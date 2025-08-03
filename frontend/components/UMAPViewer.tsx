"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { Data as PlotlyData, Layout as PlotlyLayout, Config as PlotlyConfig } from "plotly.js";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";

// Dynamically import Plot from react-plotly.js for client-side rendering
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type EmbeddingPoint = {
  x: number;
  y: number;
  label: string; // "Alive" or "Dead"
  patient_id: string;
};

type EmbeddingsDataMap = {
  [modelName: string]: EmbeddingPoint[];
};

export default function UMAPViewer({ runId }: { runId?: string }) {
  const [allEmbeddingsData, setAllEmbeddingsData] = useState<EmbeddingsDataMap>({});
  const [selectedModel, setSelectedModel] = useState<string>("global");
  const [colorMode, setColorMode] = useState<"class" | "source">("class");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return; // 🛑 Avoid fetch if runId is not yet set

    const fetchEmbeddings = async () => {
      setLoading(true);
      setError(null);
      try {
        const url = `${process.env.NEXT_PUBLIC_BACKEND_URL}/dissect/embeddings?run_id=${encodeURIComponent(runId)}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const rawResponse = await response.json();

        if (!rawResponse || typeof rawResponse !== "object" || !rawResponse.embeddings) {
          throw new Error("Invalid response: expected an object with 'embeddings' property");
        }

        const data: EmbeddingsDataMap = rawResponse.embeddings;
        setAllEmbeddingsData(data);

        if (data.global) {
          setSelectedModel("global");
        } else if (Object.keys(data).length > 0) {
          setSelectedModel(Object.keys(data)[0]);
        } else {
          setError("No UMAP data available. Ensure models are trained and saved.");
        }
      } catch (err: any) {
        setError(`Failed to load embeddings: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    fetchEmbeddings();
  }, [runId]);

  const generatePlotlyTraces = (): PlotlyData[] => {
    const traces: PlotlyData[] = [];
    const colorPalette = [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
      '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
      '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ];

    if (colorMode === "class") {
      const points = allEmbeddingsData[selectedModel] || [];
      const uniqueLabels = Array.from(new Set(points.map(p => p.label)));

      uniqueLabels.forEach((label, idx) => {
        const group = points.filter(p => p.label === label);
        traces.push({
          x: group.map(p => p.x),
          y: group.map(p => p.y),
          mode: "markers",
          type: "scattergl",
          name: label,
          text: group.map(p => `Patient: ${p.patient_id}<br>Status: ${p.label}`),
          hoverinfo: "text",
          marker: {
            size: 8,
            opacity: 0.8,
            color: colorPalette[idx % colorPalette.length],
          },
        });
      });
    } else {
      Object.keys(allEmbeddingsData).forEach((modelKey, idx) => {
        const group = allEmbeddingsData[modelKey];
        const displayName = modelKey === "global" ? "Global Model" : `Client ${modelKey.split('_')[1]}`;
        traces.push({
          x: group.map(p => p.x),
          y: group.map(p => p.y),
          mode: "markers",
          type: "scattergl",
          name: displayName,
          text: group.map(p => `Patient: ${p.patient_id}<br>Status: ${p.label}`),
          hoverinfo: "text",
          marker: {
            size: 8,
            opacity: 0.8,
            color: colorPalette[idx % colorPalette.length],
          },
        });
      });
    }

    return traces;
  };

  const layout: PlotlyLayout = {
    title: {
      text: colorMode === "source"
        ? "Patient Embedding Space (UMAP) - All Models"
        : `UMAP - ${selectedModel === "global" ? "Global Model" : `Client ${selectedModel.split('_')[1]}`}`,
    },
    hovermode: "closest",
    uirevision: true,
    xaxis: { title: { text: "UMAP Dimension 1" } },
    yaxis: { title: { text: "UMAP Dimension 2" } },
    margin: { l: 40, r: 40, b: 40, t: 80 },
    legend: {
      x: 1.02,
      y: 1,
      xanchor: "left",
      yanchor: "top",
    },
  };

  const config: Partial<PlotlyConfig> = {
    responsive: true,
    displayModeBar: false,
  };

  if (!runId) return null;
  if (loading) return <p className="text-center">Loading UMAP embeddings...</p>;
  if (error) return <p className="text-center text-red-500">{error}</p>;

  return (
    <div className="w-full max-w-4xl mx-auto mt-8 p-4 bg-white shadow-md rounded-lg">
      <h2 className="text-xl font-semibold mb-4 text-center">
        Patient Embedding Space (UMAP)
      </h2>

      <div className="mb-4 flex flex-col sm:flex-row justify-center items-center gap-4">
        <div className="flex items-center gap-2">
          <Label htmlFor="model-select">Select Model:</Label>
          <Select
            value={selectedModel}
            onValueChange={setSelectedModel}
            disabled={colorMode === "source"}
          >
            <SelectTrigger id="model-select" className="w-[180px]">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(allEmbeddingsData).map((modelKey) => (
                <SelectItem key={modelKey} value={modelKey}>
                  {modelKey === "global" ? "Global Model" : `Client ${modelKey.split("_")[1]}`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Label>Color By:</Label>
          <RadioGroup
            value={colorMode}
            onValueChange={(value: "class" | "source") => setColorMode(value)}
            className="flex gap-4"
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="class" id="color-class" />
              <Label htmlFor="color-class">Class</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="source" id="color-source" />
              <Label htmlFor="color-source">Source</Label>
            </div>
          </RadioGroup>
        </div>
      </div>

      <Plot
        data={generatePlotlyTraces()}
        layout={layout}
        config={config}
        style={{ width: "100%", height: "500px" }}
      />
    </div>
  );
}
