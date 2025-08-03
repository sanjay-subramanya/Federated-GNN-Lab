"use client";

import React, { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface BackendRoundData {
  round: number;
  global_loss: number;
  client_train: { [key: string]: number };
  client_val: { [key: string]: number };
  run_id?: string;
}

interface FLVisualizerProps {
  onComplete?: (runId: string) => void;
}

interface RoundData {
  round: number;
  global_loss: number;
  [key: string]: number;
}

export default function FLVisualizer({ onComplete }: FLVisualizerProps) {
  const [numClients, setNumClients] = useState(3);
  const [numRounds, setNumRounds] = useState(10);
  const [globalLosses, setGlobalLosses] = useState<RoundData[]>([]);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [capturedRunId, setCapturedRunId] = useState<string | null>(null);

  const handleStartTraining = async () => {
    console.log("DEBUG: Starting training with numClients:", numClients, "numRounds:", numRounds);
    setTraining(true);
    setGlobalLosses([]);
    setError(null);
    setCapturedRunId(null);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
        },
        body: JSON.stringify({
          num_clients: numClients,
          num_rounds: numRounds,
        }),
      });

      console.log("DEBUG: /train response status:", response.status, response.statusText);

      if (!response.ok) {
        const errorBody = await response.text().catch(() => "No body available");
        console.error("DEBUG: Training request failed with status:", response.status, "body:", errorBody);
        if (response.status === 404) {
          throw new Error(
            "Training endpoint not found (404). Ensure FastAPI server is running locally and /train route is correctly registered in main.py."
          );
        }
        throw new Error(`HTTP error! status: ${response.status}, body: ${errorBody}`);
      }

      // Log all response headers
      const headers: { [key: string]: string } = {};
      response.headers.forEach((value, key) => {
        headers[key] = value;
      });
      console.log("DEBUG: /train response headers:", headers);

      // Try case-insensitive header access
      const runIdFromHeader = response.headers.get("X-Run-Id") || response.headers.get("x-run-id") || null;
      console.log("DEBUG: Run ID from header:", runIdFromHeader);

      let streamRunId: string | null = null;
      if (response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            console.log("DEBUG: Training stream complete, runId:", runIdFromHeader || streamRunId || capturedRunId);
            const finalRunId = runIdFromHeader || streamRunId || `fallback-${Date.now()}`;
            setCapturedRunId(finalRunId);
            if (onComplete) {
              console.log("DEBUG: Calling onComplete with runId:", finalRunId);
              onComplete(finalRunId);
            } else {
              console.error("DEBUG: onComplete callback not provided");
              setError("Training completed but no callback to proceed");
            }
            break;
          }

          buffer += decoder.decode(value);
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const data: BackendRoundData = JSON.parse(line);
              console.log("DEBUG: Received round data:", data);
              if (data.run_id && !streamRunId) {
                streamRunId = data.run_id;
                console.log("DEBUG: Run ID from stream data:", streamRunId);
              }
              const formattedData: RoundData = {
                round: data.round,
                global_loss: data.global_loss,
              };
              Object.keys(data.client_train).forEach((clientKey, index) => {
                formattedData[`client_train_${index + 1}`] = data.client_train[clientKey] || 0;
              });
              Object.keys(data.client_val).forEach((clientKey, index) => {
                formattedData[`client_${index + 1}`] = data.client_val[clientKey] || 0;
              });
              setGlobalLosses((prev) => [...prev, formattedData]);
            } catch (error) {
              console.error("DEBUG: Error parsing line:", line, error);
            }
          }
        }
      } else {
        console.error("DEBUG: No response body received from /train");
        setError("No response body received from training endpoint");
        const fallbackRunId = `fallback-${Date.now()}`;
        setCapturedRunId(fallbackRunId);
        if (onComplete) {
          console.log("DEBUG: Calling onComplete with fallback runId:", fallbackRunId);
          onComplete(fallbackRunId);
        }
      }
    } catch (err: any) {
      console.error("DEBUG: Training error:", err.message);
      if (err.message.includes("Failed to fetch")) {
        setError(
          "Failed to connect to backend (CORS or server error). Ensure FastAPI server is running locally and /train route is registered, and CORS middleware includes allow_origins=['http://localhost:3000', FRONTEND_URL], expose_headers=['X-Run-Id']."
        );
      } else if (err.message.includes("Training endpoint not found")) {
        setError(err.message);
      } else {
        setError(`Training failed: ${err.message}`);
      }
    } finally {
      setTraining(false);
    }
  };

  const formatTooltipLabel = (value: number) => `Round ${value}`;

  const getMaxLoss = () => {
    let max = 0;
    globalLosses.forEach((data) => {
      if (data.global_loss > max) max = data.global_loss;
      for (let i = 1; i <= numClients; i++) {
        if (data[`client_train_${i}`] && data[`client_train_${i}`] > max)
          max = data[`client_train_${i}`];
        if (data[`client_${i}`] && data[`client_${i}`] > max)
          max = data[`client_${i}`];
      }
    });
    return [0, Math.ceil(Math.max(1, max))];
  };

  return (
    <div className="p-6 border rounded-lg shadow-sm bg-white">
      <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">
        🚀 Federated Learning Visualizer
      </h2>

      <div className="flex flex-col md:flex-row items-center justify-center gap-4 mb-8">
        <div className="flex items-center gap-2">
          <Label htmlFor="numClients" className="text-gray-700">
            Number of Clients:
          </Label>
          <Select
            value={String(numClients)}
            onValueChange={(val) => setNumClients(Number(val))}
            disabled={training}
          >
            <SelectTrigger className="w-[100px]">
              <SelectValue placeholder="Clients" />
            </SelectTrigger>
            <SelectContent>
              {[2, 3, 4, 5].map((n) => (
                <SelectItem key={n} value={String(n)}>
                  {n}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Label htmlFor="numRounds" className="text-gray-700">
            Number of Rounds:
          </Label>
          <Select
            value={String(numRounds)}
            onValueChange={(val) => setNumRounds(Number(val))}
            disabled={training}
          >
            <SelectTrigger className="w-[100px]">
              <SelectValue placeholder="Rounds" />
            </SelectTrigger>
            <SelectContent>
              {[5, 10, 15, 20].map((n) => (
                <SelectItem key={n} value={String(n)}>
                  {n}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Button onClick={handleStartTraining} disabled={training} className="px-6 py-2">
          {training ? "Training in progress..." : "Start Training"}
        </Button>
      </div>

      {error && (
        <p className="text-red-500 text-center mb-4">
          {error}
          <br />
          <strong>Debug Note:</strong> Check browser console (F12) and FastAPI terminal for details.
        </p>
      )}

      {capturedRunId && (
        <p className="text-gray-600 text-center mb-4">
          <strong>Run ID:</strong> {capturedRunId}
        </p>
      )}

      {globalLosses.length > 0 ? (
        <>
          <div className="mt-8 bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-medium mb-4 text-gray-800">
              📊 Global Model Loss
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={globalLosses}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" tickFormatter={formatTooltipLabel} />
                <YAxis domain={getMaxLoss()} />
                <Tooltip labelFormatter={formatTooltipLabel} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="global_loss"
                  stroke="#8884d8"
                  strokeWidth={2}
                  name="Global Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-12 bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-medium mb-4 text-gray-800">
              📈 Client Validation Losses
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={globalLosses}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" tickFormatter={formatTooltipLabel} />
                <YAxis domain={getMaxLoss()} />
                <Tooltip labelFormatter={formatTooltipLabel} />
                <Legend />
                {[...Array(numClients)].map((_, i) => (
                  <Line
                    key={`client_${i + 1}`}
                    type="monotone"
                    dataKey={`client_${i + 1}`}
                    stroke={["#00bcd4", "#4caf50", "#f44336", "#ff9800", "#9c27b0"][i % 5]}
                    strokeWidth={2}
                    name={`Client ${i + 1} Val Loss`}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-12 bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-medium mb-4 text-gray-800">
              🏋️‍♂️ Client Training Losses
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={globalLosses}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" tickFormatter={formatTooltipLabel} />
                <YAxis domain={getMaxLoss()} />
                <Tooltip labelFormatter={formatTooltipLabel} />
                <Legend />
                {[...Array(numClients)].map((_, i) => (
                  <Line
                    key={`client_train_${i + 1}`}
                    type="monotone"
                    dataKey={`client_train_${i + 1}`}
                    stroke={["#3f51b5", "#009688", "#cddc39", "#ff5722", "#607d8b"][i % 5]}
                    strokeWidth={2}
                    name={`Client ${i + 1} Train Loss`}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <p className="text-center text-gray-600 mt-8">
          Click "Start Training" to visualize federated learning progress.
          <br />
          <strong>Debug Note:</strong> Check browser console (F12) and FastAPI terminal for logs.
        </p>
      )}
    </div>
  );
}