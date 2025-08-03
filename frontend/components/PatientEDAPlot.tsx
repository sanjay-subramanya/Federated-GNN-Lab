"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const STAGE_ORDER = ["Stage I", "Stage II", "Stage III", "Stage IV"];
const STATUS_ORDER = ["Dead", "Alive"];
const STATUS_COLORS = {
  Dead: "#ff7f0e",
  Alive: "#1f77b4"
};

export default function PatientEDAPlot() {
  const [data, setData] = useState([]);
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/eda`)
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error("Failed to fetch EDA data:", err));
  }, []);

  const traces = [];
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
          showlegend: stage === STAGE_ORDER[0],
          side: status === "Dead" ? "negative" : "positive",
          width: 0.88,
          fillcolor: "rgba(0,0,0,0)", // NO fill, only outline
          line: { color: STATUS_COLORS[status], width: 3 }, // Bold outline
          opacity: 0.8,
          box: { visible: false },
          points: "all",
          pointpos: status === "Dead" ? -0.25 : 0.25,
          jitter: 0.32,
          marker: {
            size: 8,
            color: STATUS_COLORS[status],
            symbol: "circle",
            opacity: 0.8,
            line: { width: 1, color: "#222" }
          },
          scalemode: "width", // NORMALIZE all violin widths
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

  const legendAnnotations = [
    {
      xref: "paper", yref: "paper",
      x: 0.79, y: 1.14,
      showarrow: false,
      text: `<span style='display:inline-block;width:18px;height:18px;background:${STATUS_COLORS.Alive};border-radius:9px;margin-right:7px;vertical-align:middle'></span>Alive`,
      align: "left",
      font: { size: 15 }
    },
    {
      xref: "paper", yref: "paper",
      x: 0.91, y: 1.14,
      showarrow: false,
      text: `<span style='display:inline-block;width:18px;height:18px;background:${STATUS_COLORS.Dead};border-radius:9px;margin-right:7px;vertical-align:middle'></span>Dead`,
      align: "left",
      font: { size: 15 }
    }
  ];

  return (
    <Card className="mb-8 shadow-lg">
      <CardHeader>
        <CardTitle className="text-center text-xl tracking-tight font-semibold">
          🏆 Age Distribution by Tumor Stage & Survival Status
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex justify-center mb-4">
          <ToggleGroup
            type="single"
            value={filter}
            onValueChange={val =>
              setFilter(
                val === "all" || val === "dead" || val === "alive" ? val : "all"
              )
            }
          >
            <ToggleGroupItem value="all">Show All</ToggleGroupItem>
            <ToggleGroupItem value="alive">Only Alive</ToggleGroupItem>
            <ToggleGroupItem value="dead">Only Dead</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div style={{ width: "100%", minHeight: 400 }}>
          <Plot
            data={traces}
            layout={{
              font: {
                family: "Inter, Arial, sans-serif",
                size: 17,
                color: "#222"
              },
              xaxis: {
                title: "Tumor Stage",
                tickfont: { size: 16 },
                zeroline: false,
                showgrid: false,
                showline: true,
                linecolor: "#e9eaee",
                categoryorder: "array",
                categoryarray: STAGE_ORDER
              },
              yaxis: {
                title: "Patient Age",
                tickfont: { size: 16 },
                gridcolor: "#f8f8f8",
                zeroline: false,
                showline: true,
                linecolor: "#e9eaee"
              },
              violinmode: "overlay",
              violingap: 0.17,
              plot_bgcolor: "#fff8fa",
              paper_bgcolor: "#fafaff",
              showlegend: true,
              legend: {
                orientation: "h",
                x: 0.48,
                xanchor: "center",
                y: 1.09
              },
              margin: { t: 22, l: 64, r: 24, b: 54 },
              height: 440,
              hoverlabel: {
                bgcolor: "#22223b",
                font: { color: "#f2f2f2", size: 15 }
              },
              annotations: legendAnnotations
            }}
            config={{
              responsive: true,
              displayModeBar: false
            }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
