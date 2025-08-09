"use client";

import React, { useEffect, useState, useCallback } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { NEXT_PUBLIC_BACKEND_URL } from "@/lib/config";

// Matches the backend's FeatureOverlap Pydantic model
interface FeatureOverlap {
  overlap_percentage: number;
  common_features: string[];
}

// Matches the backend's FeatureImportanceEntry Pydantic model
interface FeatureImportanceEntry {
  feature_name: string;
  importance: number;
}

// Matches the backend's FeatureImportanceResponse Pydantic model, including new overlap field
interface FeatureImportanceResponse {
  model_name: string;
  feature_importances: FeatureImportanceEntry[];
  overlap_with_global?: FeatureOverlap; 
}

export default function FeatureImportanceViewer({ runId, onLoadComplete }: {
    runId?: string;
    onLoadComplete?: () => void;
  }) {
  const [selectedModel, setSelectedModel] = useState<string>("global");
  const [featureImportances, setFeatureImportances] = useState<FeatureImportanceEntry[]>([]);
  const [overlapData, setOverlapData] = useState<FeatureOverlap | null>(null); 
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [numClients, setNumClients] = useState<number | null>(null);

  useEffect(() => {
    const fetchTrainMetadata = async () => {
      if (!runId) return;
      try {
        const res = await fetch(`${NEXT_PUBLIC_BACKEND_URL}/train-metadata?run_id=${runId}`);
        const data = await res.json();
        setNumClients(data.num_clients);
      } catch (err) {
        console.warn("[FeatureImportanceViewer] Could not load training metadata:", err);
      }
    };
    fetchTrainMetadata();
  }, [runId]);

  const availableModels = [
    { value: "global", label: "Global Model" },
    ...(numClients ? Array.from({ length: numClients }, (_, i) => ({
      value: `client_${i + 1}`,
      label: `Client ${i + 1}`,
    })) : []),
  ];
  
  const fetchFeatureImportances = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const baseUrl = `${NEXT_PUBLIC_BACKEND_URL}/dissect/feature-importance`;
      const url = new URL(baseUrl);
      url.searchParams.append("model_name", selectedModel);
      url.searchParams.append("top_k", "20");
      if (runId) url.searchParams.append("run_id", runId);

      const response = await fetch(url.toString());
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: FeatureImportanceResponse = await response.json();
      setFeatureImportances(data.feature_importances);
      setOverlapData(data.overlap_with_global || null);
      if (onLoadComplete) onLoadComplete();
    } catch (e: any) {
      console.error("Failed to fetch feature importances:", e);
      setError(`Failed to load feature importances: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedModel, runId, onLoadComplete]);

  useEffect(() => {
    fetchFeatureImportances();
  }, [fetchFeatureImportances]); // Re-fetch when selectedModel changes

  // Custom Tick Formatter for long feature names
  const formatFeatureNameTick = (tickItem: string) => {
    if (tickItem.length > 15) {
      return tickItem.substring(0, 12) + "..."; // Truncate long names
    }
    return tickItem;
  };

  return (
    <div className="w-full max-w-5xl mx-auto mt-8 p-4 bg-white shadow-md rounded-lg">
      {/* Model Selection Dropdown */}
      <div className="mb-6 flex justify-center items-center gap-4">
        <Label htmlFor="feature-model-select" className="font-medium text-gray-900">
          Select Model:
        </Label>
        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger id="feature-model-select" className="w-[200px] text-gray-900">
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            {availableModels.map((model) => (
              <SelectItem key={model.value} value={model.value}>
                {model.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Content Area: Loading, Error, Charts */}
      {loading ? (
        <div className="space-y-4">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-[300px] w-full" />
        </div>
      ) : error ? (
        <p className="text-red-500 text-center">{error}</p>
      ) : (
        <div className="grid grid-cols-1 gap-8"> {/* NEW: Grid layout for side-by-side */}
          {/* Feature Importance Bar Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-center">
                Top 20 Features for {selectedModel === 'global' ? 'Global Model' : `Client ${selectedModel.split('_')[1]} Model`}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {featureImportances.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart
                    data={featureImportances}
                    margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis
                      type="category"
                      dataKey="feature_name"
                      width={150}
                      tickFormatter={formatFeatureNameTick}
                      interval={0}
                    />
                    <Tooltip
                      formatter={(value: number, name: string, props: any) => [
                        value.toFixed(4),
                        "Importance",
                      ]}
                      labelFormatter={(label: string) => `Feature: ${label}`}
                    />
                    <Legend />
                    <Bar dataKey="importance" fill="#8884d8" name="Importance Score" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-center text-gray-500">No feature importance data available.</p>
              )}
            </CardContent>
          </Card>

          {/* Overlap with Global Model Panel */}
          {selectedModel !== "global" && overlapData && (
            <Card className="col-span-1">
              <CardHeader>
                <CardTitle className="text-center">
                  Overlap with Global Model (Top 20)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center mb-4">
                  <p className="text-lg font-semibold">
                    Overlap Percentage: <span className="text-blue-600 text-2xl">{overlapData.overlap_percentage}%</span>
                  </p>
                </div>
                <h4 className="font-medium mb-2">Common Features:</h4>
                {overlapData.common_features.length > 0 ? (
                  <ul className="list-disc list-inside max-h-40 overflow-y-auto">
                    {overlapData.common_features.map((feature, index) => (
                      <li key={index} className="text-sm">{feature}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-500 text-sm">No common features found in the top 20.</p>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}