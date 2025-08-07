"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false }) as React.FC<any>;

interface RoundDivergenceData {
  round: number;
  global_loss: number;
  client_divergence: {
    [clientId: string]: {
      [layerName: string]: number;
    };
  };
}

export default function DivergenceViewer({ runId, onLoadComplete}: {
    runId?: string;
    onLoadComplete?: () => void;
  }) {
  const [divergenceHistory, setDivergenceHistory] = useState<RoundDivergenceData[] | null>(null);
  const [viewMode, setViewMode] = useState<"line" | "heatmap" | "table">("line");
  const [selectedRoundIndex, setSelectedRoundIndex] = useState<number>(-1);
  const [scaleMode, setScaleMode] = useState<"fixed" | "dynamic">("fixed");
  const [deltaMode, setDeltaMode] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [numClients, setNumClients] = useState<number | null>(null);
  const [numRounds, setNumRounds] = useState<number | null>(null);

  useEffect(() => {
    const fetchMetadata = async () => {
      if (!runId) return;
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/train-metadata?run_id=${runId}`);
        const data = await res.json();
        setNumClients(data.num_clients);
        setNumRounds(data.num_rounds);
      } catch (err) {
        console.warn("[DivergenceViewer] Failed to fetch training metadata:", err);
      }
    };
    fetchMetadata();
  }, []);

  const fetchData = useCallback(async () => {
  if (!runId) return;  // ✅ Prevent premature call

  try {
    const baseUrl = `${process.env.NEXT_PUBLIC_BACKEND_URL}/dissect/divergence-history`;
    const url = `${baseUrl}?run_id=${encodeURIComponent(runId)}`;
    const response = await fetch(url);
    const json = await response.json();

    if (!Array.isArray(json)) {
      console.warn("[DivergenceViewer] Expected array but got:", json);
      return;
    }

    setDivergenceHistory(json);
    onLoadComplete?.();
  } catch (err) {
    console.warn("[DivergenceViewer] Failed to fetch:", err);
  }
  }, [runId]);

  useEffect(() => {
  if (!runId) return; // Wait for runId before attempting fetch
  fetchData();
  }, [fetchData, runId]);

  useEffect(() => {
  // Clear any existing interval first
    if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
    }
    if (isPlaying && divergenceHistory) {
        playIntervalRef.current = setInterval(() => {
        setSelectedRoundIndex(prev => {
            const next = prev + 1;
            if (next >= divergenceHistory.length) {
            setIsPlaying(false); // stop
            return prev;
            }
            return next;
        });
        }, 1000);
    }
    // Clean up on unmount
    return () => {
        if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
        }
    };
    }, [isPlaying, divergenceHistory]);

  if (!divergenceHistory) {
    return (
      <Card>
      <CardHeader><CardTitle>Client Divergence Tracker</CardTitle></CardHeader>
      <CardContent>
        <Skeleton className="h-64 w-full" />
        {numClients !== null && numRounds !== null && (
          <p className="text-muted-foreground text-center mt-4">
            Expecting {numClients} clients over {numRounds} rounds…
          </p>
        )}
      </CardContent>
      </Card>
    );
  }

  const latestIndex = divergenceHistory.length - 1;
  const roundIndex = selectedRoundIndex >= 0 ? selectedRoundIndex : latestIndex;
  const roundData = divergenceHistory[roundIndex];
  const rounds = divergenceHistory.map(d => d.round);
  const clientIds = Object.keys(roundData.client_divergence);
  const layerNames = Object.keys(roundData.client_divergence[clientIds[0]]);

  const computeInsights = () => {
    let max = -Infinity, min = Infinity;
    let maxClient = "", maxLayer = "";
    let total = 0, count = 0, converged = 0;

    for (const client of clientIds) {
      for (const layer of layerNames) {
        const val = roundData.client_divergence[client]?.[layer];
        if (val > max) {
          max = val;
          maxClient = client;
          maxLayer = layer;
        }
        if (val < min) min = val;
        total += val;
        count += 1;
        if (val < 0.01) converged += 1;
      }
    }

    return {
      avg: total / count,
      maxClient,
      maxLayer,
      maxVal: max,
      convergedCount: converged,
    };
  };

  const insights = computeInsights();

  const getHeatmapZ = (): number[][] => {
    if (deltaMode && roundIndex > 0) {
      const prev = divergenceHistory[roundIndex - 1];
      return clientIds.map(client =>
        layerNames.map(layer =>
          roundData.client_divergence[client]?.[layer] - prev.client_divergence[client]?.[layer] || 0
        )
      );
    } else {
      return clientIds.map(client =>
        layerNames.map(layer =>
          roundData.client_divergence[client]?.[layer] ?? NaN
        )
      );
    }
  };

  const heatmapZ = getHeatmapZ();
  const flatZ = heatmapZ.flat();
  const zMin = scaleMode === "fixed" ? (deltaMode ? -0.01 : 0.0) : Math.min(...flatZ);
  const zMax = scaleMode === "fixed" ? (deltaMode ? 0.01 : 0.05) : Math.max(...flatZ);

  const lineTraces = [
    {
      x: rounds,
      y: divergenceHistory.map(d => d.global_loss),
      name: "Global Loss",
      mode: "lines+markers",
      line: { color: "blue" },
      yaxis: "y1",
    },
    ...clientIds.map(client => {
      const avgD = divergenceHistory.map(d => {
        const vals = Object.values(d.client_divergence[client] || {});
        return vals.reduce((a, b) => a + b, 0) / vals.length;
      });
      return {
        x: rounds,
        y: avgD,
        name: `${client} Avg Div`,
        mode: "lines+markers",
        yaxis: "y2",
      };
    }),
  ];

  const HeatmapLegend = React.memo(({ zMin, zMax }: { zMin: number, zMax: number }) => (
    <div className="flex flex-col text-sm text-muted-foreground ml-2">
        <Label className="mb-2">Legend:</Label>
        <div className="h-40 w-6 bg-gradient-to-b from-yellow-50 via-orange-400 to-red-700 border" />
        <div className="flex flex-col text-center mt-1">
        <span>{zMax.toFixed(3)}</span>
        <span className="text-xs text-gray-400">Divergence</span>
        <span>{zMin.toFixed(3)}</span>
        </div>
    </div>
    ));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Client Divergence Tracker</CardTitle>
        <div className="text-sm text-muted-foreground mt-2 space-y-1">
          <div>📅 <b>Round:</b> {roundData.round}</div>
          <div>🔥 <b>Most Divergent:</b> {insights.maxClient} - {insights.maxLayer} ({insights.maxVal.toFixed(4)})</div>
          <div>📊 <b>Avg Divergence:</b> {insights.avg.toFixed(4)}</div>
          <div>✅ <b>Clients &lt; 0.01:</b> {insights.convergedCount}/{(numClients ?? clientIds.length) * layerNames.length}</div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-4 items-center">
          <Label>View:</Label>
          <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as any)}>
            <TabsList>
              <TabsTrigger value="line">📈 Line</TabsTrigger>
              <TabsTrigger value="heatmap">🔥 Heatmap</TabsTrigger>
              <TabsTrigger value="table">📋 Table</TabsTrigger>
            </TabsList>
          </Tabs>
        
        <div className="text-xs text-muted-foreground mt-1 space-y-1">
            <div>🔁 <b>Delta Mode:</b> Highlights how much each client-layer's divergence <i>changed</i> from the previous round. Red = drifted more, Blue = converged.</div>
            <div>📏 <b>Scale Mode:</b> 
                <ul className="list-disc pl-4">
                <li><b>Fixed:</b> Uses same range (0–0.05) across all rounds — good for global comparison.</li>
                <li><b>Dynamic:</b> Auto-scales based on each round’s actual divergence — good for detail per round.</li>
                </ul>
            </div>
            <div>▶️ <b>Play:</b> Animates round-by-round heatmap. <i>Stops at last round automatically.</i></div>
            </div>

          {viewMode !== "line" && (
            <>
              <Label>Round:</Label>
              <Select value={String(roundIndex)} onValueChange={v => setSelectedRoundIndex(parseInt(v))}>
                <SelectTrigger className="w-[100px]"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {divergenceHistory.map((d, i) => (
                    <SelectItem key={i} value={String(i)}>R{d.round}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={() => {
                    if (isPlaying) {
                    setIsPlaying(false); // pause
                    } else {
                    if (selectedRoundIndex >= divergenceHistory.length - 1) {
                        setSelectedRoundIndex(0); // restart from beginning
                    }
                    setIsPlaying(true); // play
                    }
                }}
                variant="outline"
                >
                {isPlaying ? "⏸ Pause" : "▶️ Play"}
                </Button>
              <Button variant="ghost" onClick={() => setDeltaMode(!deltaMode)}>
                {deltaMode ? "🟢 Delta Mode: ON" : "⚪ Delta Mode: OFF"}
              </Button>
              <Tabs value={scaleMode} onValueChange={(v) => setScaleMode(v as any)}>
                <TabsList>
                  <TabsTrigger value="fixed">Fixed (0–0.05)</TabsTrigger>
                  <TabsTrigger value="dynamic">Dynamic</TabsTrigger>
                </TabsList>
              </Tabs>
            </>
          )}
        </div>

        {viewMode === "line" && (
          <Plot
            data={lineTraces}
            layout={{
              xaxis: { title: "Round", dtick: 1 },
              yaxis: { title: "Global Loss", side: "left" },
              yaxis2: { title: "Client Div", overlaying: "y", side: "right" },
              height: 400,
              legend: { orientation: "h" },
            }}
            config={{ responsive: true }}
          />
        )}

        {viewMode === "heatmap" && (
        <div className="flex gap-4 items-start">
            <div className="w-full max-w-[calc(100%-60px)]">
            <Plot
                data={[{
                z: heatmapZ,
                x: layerNames,
                y: clientIds,
                type: "heatmap",
                colorscale: deltaMode ? "RdBu" : "YlOrRd",
                zmin: zMin,
                zmax: zMax,
                }]}
                layout={{
                title: `Client vs Layer ${deltaMode ? "ΔDivergence" : "Divergence"} (Round ${roundData.round})`,
                height: 400,
                margin: { t: 40, b: 80 },
                uirevision: 'heatmap-fixed-layout',
                }}
                config={{ responsive: true }}
            />
            </div>
            <HeatmapLegend zMin={zMin} zMax={zMax} />
        </div>
        )}

        {viewMode === "table" && (
          <div className="overflow-auto max-h-[400px]">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Client</TableHead>
                  {layerNames.map(layer => (
                    <TableHead key={layer}>{layer}</TableHead>
                  ))}
                  <TableHead>Avg</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {clientIds.map(client => {
                  const vals = layerNames.map(l => roundData.client_divergence[client]?.[l] ?? NaN);
                  const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
                  return (
                    <TableRow key={client}>
                      <TableCell>{client}</TableCell>
                      {vals.map((v, i) => <TableCell key={i}>{v.toFixed(4)}</TableCell>)}
                      <TableCell className="font-semibold">{avg.toFixed(4)}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
