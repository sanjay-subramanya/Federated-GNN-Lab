"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { NEXT_PUBLIC_BACKEND_URL } from "@/lib/config";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false }) as React.FC<any>;

const STAGE_ORDER = ["Stage I", "Stage II", "Stage III", "Stage IV"];
const STATUS_ORDER = ["Dead", "Alive"] as const;
const STATUS_COLORS: Record<(typeof STATUS_ORDER)[number], string> = {
  Dead: "#FF6B6B",
  Alive: "#4ECDC4"
};

interface PatientEntry {
  id: string;
  stage: string;
  age: number;
  status: "Alive" | "Dead";
}

interface ViolinTrace extends Partial<Plotly.PlotData> {
  side?: "positive" | "negative";
  box?: { visible: boolean };
  pointpos?: number;
  jitter?: number;
  points?: "all" | "outliers" | "suspectedoutliers" | false;
  spanmode?: "soft" | "hard";
  scalemode?: "width" | "count";
}

export default function PatientEDAPlot() {
  const [data, setData] = useState<PatientEntry[]>([]);
  const [filter, setFilter] = useState<"all" | "alive" | "dead">("all");

  useEffect(() => {
    fetch(`${NEXT_PUBLIC_BACKEND_URL}/eda`)
      .then((res) => res.json())
      .then((json: PatientEntry[]) => setData(json))
      .catch((err) => console.error("Failed to fetch EDA data:", err));
  }, []);

  const traces: ViolinTrace[] = [];

  STAGE_ORDER.forEach((stage) => {
    STATUS_ORDER.forEach((status) => {
      let filtered = data.filter(
        (d) => d.stage === stage && d.status === status
      );
      if (filter !== "all" && filter !== status.toLowerCase()) filtered = [];
      if (filtered.length) {
        traces.push({
          type: "violin",
          y: filtered.map((d) => d.age),
          x: filtered.map(() => stage),
          name: status,
          legendgroup: status,
          showlegend: false, 
          side: status === "Dead" ? "negative" : "positive",
          width: 0.88,
          fillcolor: "rgba(0,0,0,0)",
          line: { color: STATUS_COLORS[status], width: 3 },
          opacity: 0.9,
          box: { visible: false },
          points: "all",
          pointpos: status === "Dead" ? -0.25 : 0.25,
          jitter: 0.32,
          marker: {
            size: 8,
            color: STATUS_COLORS[status],
            symbol: "circle",
            opacity: 0.85,
            line: { width: 1, color: "#1A1A2E" }
          },
          scalemode: "width",
          spanmode: "soft",
          text: filtered.map(
            d =>
              `Patient: ${d.id}<br>Status: ${d.status}<br>Age: ${d.age}`
          ),
          hoverinfo: "text"
        });
      }
    });
  });

  return (
    <div className="w-full max-w-5xl mx-auto mt-8 flex flex-col justify-center items-center">
      <div className="text-center text-2xl font-semibold text-[#00C1D5] tracking-tight pb-4">
        🧬 Age Distribution by Tumor Stage & Survival Status
      </div>
      <div className="w-full px-8 pb-8">
        <div className="flex justify-center mb-6">
          <ToggleGroup
            type="single"
            value={filter}
            onValueChange={(val) =>
              setFilter(
                val === "all" || val === "dead" || val === "alive" ? val : "all"
              )
            }
            className="bg-[#2D3748] rounded-lg p-1"
          >
            <ToggleGroupItem
              value="all"
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                filter === "all"
                  ? "bg-[#00C1D5] text-white"
                  : "text-[#CBD5E1] hover:bg-[#3B4A5A]"
              }`}
            >
              Show All
            </ToggleGroupItem>
            <ToggleGroupItem
              value="alive"
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                filter === "alive"
                  ? "bg-[#00C1D5] text-white"
                  : "text-[#CBD5E1] hover:bg-[#3B4A5A]"
              }`}
            >
              Only Alive
            </ToggleGroupItem>
            <ToggleGroupItem
              value="dead"
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
                filter === "dead"
                  ? "bg-[#00C1D5] text-white"
                  : "text-[#CBD5E1] hover:bg-[#3B4A5A]"
              }`}
            >
              Only Dead
            </ToggleGroupItem>
          </ToggleGroup>
        </div>

        {/* PLOT + HTML LEGEND */}
        <div className="w-full" style={{ minHeight: 450 }}>
          {/* Use column on small screens and row on md+ so legend stacks nicely */}
          <div className="flex flex-col md:flex-row items-start justify-center gap-6">
            {/* Plot column (centered, constrained width so it appears centered visually) */}
            <div className="flex-1 w-full max-w-4xl">
              <div className="mx-auto" style={{ maxWidth: 900 }}>
                <Plot
                  data={traces}
                  layout={{
                    font: {
                      family: "Inter, sans-serif",
                      size: 15,
                      color: "#E0E7EB"
                    },
                    xaxis: {
                      title: {
                        text: "Tumor Stage",
                        font: { size: 16, color: "#E0E7EB" }
                      },
                      tickfont: { size: 14, color: "#E0E7EB" },
                      zeroline: false,
                      showgrid: false,
                      showline: true,
                      linecolor: "#4B5563",
                      categoryorder: "array",
                      categoryarray: STAGE_ORDER,
                      showspikes: false,
                      spikedistance: -1
                    },
                    yaxis: {
                      title: {
                        text: "Patient Age",
                        font: { size: 16, color: "#E0E7EB" }
                      },
                      tickfont: { size: 14, color: "#E0E7EB" },
                      zeroline: false,
                      showgrid: false,
                      showline: true,
                      linecolor: "#4B5563",
                      showspikes: false,
                      spikedistance: -1
                    },
                    violinmode: "overlay",
                    violingap: 0.17,
                    plot_bgcolor: "#1F2937",
                    paper_bgcolor: "#1F2937",
                    showlegend: false,
                    margin: { t: 40, l: 180, r: 10, b: 70 },
                    height: 450,
                    hovermode: "closest",
                    hoverlabel: {
                      bgcolor: "#2D3748",
                      font: { color: "#E0E7EB", size: 14, family: "Inter, sans-serif" },
                      bordercolor: "#4B5563"
                    },
                    autosize: true
                  }}
                  config={{
                    responsive: true,
                    displayModeBar: false
                  }}
                  style={{ width: "100%", height: "100%" }}
                />
              </div>
            </div>

            {/* HTML legend */}
            <div className="w-full md:w-36 flex-shrink-0 flex flex-col justify-center items-start md:items-start">
              {/* <div className="text-base text-[#E0E7EB] mb-3 font-medium">Legend</div> */}

              <div className="flex items-center gap-3 mb-2">
                <span
                  style={{
                    width: 14,
                    height: 14,
                    background: STATUS_COLORS["Dead"],
                    display: "inline-block",
                    borderRadius: 4,
                    border: "1px solid #1A1A2E"
                  }}
                />
                <span className="text-sm text-[#E0E7EB]">Dead</span>
              </div>

              <div className="flex items-center gap-3">
                <span
                  style={{
                    width: 14,
                    height: 14,
                    background: STATUS_COLORS["Alive"],
                    display: "inline-block",
                    borderRadius: 4,
                    border: "1px solid #1A1A2E"
                  }}
                />
                <span className="text-sm text-[#E0E7EB]">Alive</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
