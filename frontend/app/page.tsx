"use client";

import React, { useState, useEffect, useCallback, Component, ErrorInfo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Select from "react-select";
import { motion, AnimatePresence } from "framer-motion";
import PatientEDAPlot from "@/components/PatientEDAPlot";
import FLVisualizer from "@/components/FLVisualizer";
import UMAPViewer from "@/components/UMAPViewer";
import FeatureImportanceViewer from "@/components/FeatureImportanceViewer";
import DivergenceViewer from "@/components/DivergenceViewer";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { ChevronDown, ChevronUp  } from "lucide-react";
import { NEXT_PUBLIC_BACKEND_URL } from "@/lib/config";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  componentName: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`Error in ${this.props.componentName}:`, error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="text-red-400 p-4">
          <p>Error rendering {this.props.componentName}: {this.state.error?.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function HomePage() {
  const [patients, setPatients] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [result, setResult] = useState<{
    patient_id: string;
    prediction: string;
    confidence: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trainingDone, setTrainingDone] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  const [pollingAttempts, setPollingAttempts] = useState(0);
  const maxPollingAttempts = 24; // 120 seconds at 5000ms interval
  const [showDivergence, setShowDivergence] = useState(false);
  const [showImportance, setShowImportance] = useState(false);
  const [showUMAP, setShowUMAP] = useState(false);

  useEffect(() => {
    console.log("DEBUG: Initializing HomePage, fetching patients (check browser console)");
    fetch(`${NEXT_PUBLIC_BACKEND_URL}/patients`)
      .then((res) => res.json())
      .then((data) => {
        console.log("DEBUG: Patients fetched:", data.patient_ids);
        setPatients(data.patient_ids);
      })
      .catch((err) => {
        console.error("DEBUG: Failed to load patient list:", err);
        setError("Failed to load patient list.");
      });

    return () => {
      if (pollingInterval) {
        console.log("DEBUG: Cleaning up polling interval on unmount");
        clearInterval(pollingInterval);
      }
    };
  }, []);

  useEffect(() => {
    const handleUnload = () => {
      if (currentRunId) {
        console.log("DEBUG: Sending delete-run request for runId:", currentRunId, "at", new Date().toISOString());
        const data = JSON.stringify({ run_id: currentRunId, runId: currentRunId });
        const blob = new Blob([data], { type: "application/json" });
        navigator.sendBeacon(`${NEXT_PUBLIC_BACKEND_URL}/delete-run`, blob);
      } else {
        console.log("DEBUG: No currentRunId, skipping delete-run request at", new Date().toISOString());
      }
    };

    window.addEventListener("beforeunload", handleUnload);
    return () => {
      console.log("DEBUG: Removing beforeunload event listener");
      window.removeEventListener("beforeunload", handleUnload);
    };
  }, [currentRunId]);

  const handlePredict = async () => {
    if (selectedIndex === null) return;
    const patient_id = patients[selectedIndex];
    console.log("DEBUG: Starting prediction for patient_id:", patient_id);
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${NEXT_PUBLIC_BACKEND_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id }),
      });
      if (!res.ok) throw new Error(`Prediction failed: status ${res.status}`);
      const data = await res.json();
      console.log("DEBUG: Prediction result:", data);
      setResult(data);
    } catch (err) {
      console.error("DEBUG: Prediction error:", err);
      setError("Failed to fetch prediction or explanation.");
    } finally {
      setLoading(false);
    }
  };

  const checkAnalysisReady = async (runId: string) => {
    if (!runId) {
      console.error("DEBUG: No runId provided for analysis check");
      setError("No runId provided for analysis");
      return false;
    }
    console.log("DEBUG: Checking analysis readiness for runId:", runId);
    try {
      const url = `${NEXT_PUBLIC_BACKEND_URL}/dissect/status?run_id=${encodeURIComponent(runId)}`;
      const res = await fetch(url);
      if (!res.ok) {
        console.error("DEBUG: /dissect/status failed:", res.status, res.statusText);
        setError("Failed to check analysis readiness");
        return false;
      }
      const data = await res.json();
      console.log("DEBUG: /dissect/status response:", data);

      if (data.ready) {
        return true;
      } else {
        console.log("DEBUG: Analysis not ready yet");
        return false;
      }
    } catch (err) {
      console.error("DEBUG: Error in checkAnalysisReady:", err);
      setError("Network error while checking analysis readiness");
      return false;
    }
  };

  const handleTrainingComplete = useCallback(async (runId: string) => {
    console.log("DEBUG: handleTrainingComplete called with runId:", runId);
    if (!runId) {
      console.error("DEBUG: Received empty runId");
      setError("Training completed but no runId received");
      return;
    }
    console.log("DEBUG: Setting trainingDone=true, currentRunId:", runId);
    setCurrentRunId(runId);
    setTrainingDone(true);
    setPollingAttempts(0);
    setShowAnalysis(false);
    setError(null);

    if (pollingInterval) {
      console.log("DEBUG: Clearing existing polling interval");
      clearInterval(pollingInterval);
    }

    const checkAndShow = async () => {
      console.log(`DEBUG: Polling attempt ${pollingAttempts + 1} for runId: ${runId}`);
      const isReady = await checkAnalysisReady(runId);
      setPollingAttempts((prev) => prev + 1);
      if (isReady) {
        console.log("DEBUG: All analysis endpoints ready, setting showAnalysis to true");
        setShowAnalysis(true);
        if (pollingInterval) {
          clearInterval(pollingInterval);
          setPollingInterval(null);
        }
        return true;
      }
      if (pollingAttempts + 1 >= maxPollingAttempts) {
        console.error("DEBUG: Max polling attempts reached, stopping polling");
        setError(
          `Failed to load analysis visualizations after ${maxPollingAttempts} attempts. Ensure /dissect/embeddings, /dissect/feature-importance, and /dissect/divergence-history endpoints are available for runId=${runId}.`
        );
        setTrainingDone(false);
        clearInterval(pollingInterval!);
        setPollingInterval(null);
        return true;
      }
      console.log("DEBUG: Analysis endpoints not ready yet");
      return false;
    };

    console.log("DEBUG: Attempting immediate analysis check");
    if (await checkAndShow()) {
      console.log("DEBUG: Immediate check succeeded, no polling needed");
      return;
    }

    console.log("DEBUG: Starting polling for analysis readiness");
    const interval = setInterval(async () => {
      if (await checkAndShow()) {
        console.log("DEBUG: Polling succeeded, clearing interval");
        clearInterval(interval);
        setPollingInterval(null);
      }
    }, 5000);
    setPollingInterval(interval);
  }, [pollingAttempts, pollingInterval]);

  const handleRetryPolling = () => {
    console.log("DEBUG: Retrying polling for runId:", currentRunId);
    setError(null);
    setPollingAttempts(0);
    handleTrainingComplete(currentRunId || "");
  };

  const handleForceShowAnalysis = () => {
    console.log("DEBUG: Forcing showAnalysis to true for debugging");
    setShowAnalysis(true);
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  };

  const patientOptions = patients.map((pid, index) => ({
    value: index,
    label: `Patient #${index + 1}`,
  }));

  useEffect(() => {
    console.log("DEBUG: State update:", { trainingDone, showAnalysis, currentRunId, pollingAttempts });
  }, [trainingDone, showAnalysis, currentRunId, pollingAttempts]);

  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div className="flex flex-col items-center min-h-screen py-12 px-4 bg-[#1A1A2E] text-[#E0E7EB] font-sans overflow-x-hidden">
      <h1 className="text-5xl font-bold mb-4 text-[#00C1D5] leading-tight text-center">
        🌐 Federated Learning Lab
      </h1>
      <p className="text-xl text-[#CBD5E1] opacity-90 max-w-2xl text-center mb-10">
        A dynamic platform for exploring and visualizing FL model training and insights.
        <br />
        {/* <strong>Debug Note:</strong> Check the browser console (F12) for detailed logs. */}
      </p>

      <div className="w-full max-w-5xl flex flex-col gap-12 mt-16">

        <Collapsible defaultOpen={false} className="w-full" onOpenChange={(open) => setIsOpen(open)}>
          <CollapsibleTrigger className="flex items-center justify-between w-full bg-[#1F2937] p-6 rounded-xl shadow-lg border border-gray-700/50 text-[#00C1D5] hover:bg-[#2D3748] transition-colors duration-200">
            <h2 className="text-2xl font-semibold text-[#00C1D5]">Explore Dataset Summary</h2>
            <span className="text-[#00C1D5]">
              {isOpen ? (
            <ChevronUp className="h-6 w-6 transition-transform duration-200" />
          ) : (
            <ChevronDown className="h-6 w-6 transition-transform duration-200" />
          )}
            </span>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-4 bg-[#1F2937] p-8 rounded-xl shadow-lg border border-gray-700/50">
            <p className="text-[#CBD5E1] text-base mb-6">
              The TCGA-BRCA dataset, sourced from The Cancer Genome Atlas, is a rich collection of breast cancer data. This visualization leverages two key components: protein expression data, detailing molecular profiles of breast invasive carcinoma, and phenotype data, capturing clinical features like patient demographics, tumor stages, and survival outcomes. Together, these subsets offer insights into the molecular and clinical landscape of breast cancer, supporting exploration of disease patterns and potential biomarkers.
            </p>
            <div className="w-full max-w-3xl flex flex-col gap-6 p-6 rounded-xl bg-[#1F2937] shadow-lg border border-gray-700/50 mb-12 mx-auto">
              <div className="mb-4">
                <label htmlFor="patient-select" className="block text-[#CBD5E1] text-sm font-medium mb-2">
                  Select Patient:
                </label>
                <Select
                  id="patient-select"
                  options={patientOptions}
                  onChange={(selected) => setSelectedIndex(selected?.value ?? null)}
                  placeholder="Search or select a patient..."
                  isSearchable
                  styles={{
                    control: (provided, state) => ({
                      ...provided,
                      backgroundColor: '#2D3748',
                      borderColor: state.isFocused ? '#00C1D5' : '#4B5563',
                      color: '#E0E7EB',
                      boxShadow: state.isFocused ? '0 0 0 1px #00C1D5' : 'none',
                      '&:hover': {
                        borderColor: '#00C1D5',
                      },
                    }),
                    singleValue: (provided) => ({
                      ...provided,
                      color: '#E0E7EB',
                    }),
                    input: (provided) => ({
                      ...provided,
                      color: '#E0E7EB',
                    }),
                    placeholder: (provided) => ({
                      ...provided,
                      color: '#9CA3AF',
                    }),
                    menu: (provided) => ({
                      ...provided,
                      backgroundColor: '#2D3748',
                      borderColor: '#4B5563',
                    }),
                    option: (provided, state) => ({
                      ...provided,
                      backgroundColor: state.isFocused
                        ? 'rgba(0, 193, 213, 0.2)'
                        : state.isSelected
                        ? '#00C1D5'
                        : 'transparent',
                      color: state.isSelected ? 'white' : '#E0E7EB',
                      '&:active': {
                        backgroundColor: 'rgba(0, 193, 213, 0.3)',
                      },
                    }),
                    menuList: (provided) => ({
                      ...provided,
                      maxHeight: "200px",
                      overflowY: "auto",
                      '::-webkit-scrollbar': {
                        width: '8px',
                      },
                      '::-webkit-scrollbar-track': {
                        background: '#1F2937',
                        borderRadius: '10px',
                      },
                      '::-webkit-scrollbar-thumb': {
                        background: '#00C1D5',
                        borderRadius: '10px',
                      },
                      '::-webkit-scrollbar-thumb:hover': {
                        background: '#00A3B7',
                      },
                    }),
                  }}
                />
              </div>

              <Button
                onClick={handlePredict}
                disabled={selectedIndex === null || loading}
                className={`w-full py-3 text-lg font-semibold rounded-md transition-all duration-300
                  ${loading || selectedIndex === null ? 'bg-gray-600 text-[#CBD5E1] cursor-not-allowed' : 'bg-[#00C1D5] hover:bg-[#00A3B7] text-white'}`}
              >
                {loading ? "Predicting..." : "Check Survival Status"}
              </Button>

              {error && (
                <div className="text-red-400 text-sm mt-2 text-center">
                  <p>{error}</p>
                  {error.includes("Failed to load analysis visualizations") && currentRunId && (
                    <Button
                      onClick={handleRetryPolling}
                      className="mt-2 bg-[#00C1D5] hover:bg-[#00A3B7] text-white"
                    >
                      Retry Loading Visualizations
                    </Button>
                  )}
                </div>
              )}

              {result && (
                <Card className="mt-4 bg-[#2D3748] border border-gray-600/50 text-[#E0E7EB] shadow-md">
                  <CardContent className="p-6">
                    <h3 className="text-xl font-semibold mb-3 text-[#00C1D5]">Result</h3>
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-md">Patient ID:</p>
                      <span className="font-mono text-[#00C1D5] text-lg">{result.patient_id}</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-md">Status:</p>
                      <Badge
                        className={`text-lg px-3 py-1 font-semibold
                        ${result.prediction === "Alive" ? 'bg-green-600/80 hover:bg-green-700/80 text-white' : 'bg-red-600/80 hover:bg-red-700/80 text-white'}`}
                      >
                        {result.prediction}
                      </Badge>
                    </div>
                    {/* <div className="flex justify-between items-center"> */}
                      {/* <p className="text-md">Confidence:</p>
                      <span className="text-lg text-[#FFA000]">{result.confidence.toFixed(2)}%</span> */}
                    {/* </div> */}
                  </CardContent>
                </Card>
              )}
            </div>

            <motion.div
              className="bg-[#1F2937] p-8 rounded-xl shadow-lg border border-gray-700/50"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              {/* <h2 className="text-2xl font-semibold mb-6 text-[#00C1D5]">Dataset Overview</h2> */}
              <PatientEDAPlot />
            </motion.div>
          </CollapsibleContent>
        </Collapsible>

        <motion.div
          className="bg-[#1F2937] p-8 rounded-xl shadow-lg border border-gray-700/50"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h2 className="text-2xl font-semibold mb-6 text-[#00C1D5]">Begin Analysis Journey</h2>
          <FLVisualizer onComplete={handleTrainingComplete} />
        </motion.div>

        <AnimatePresence>
          {trainingDone && !showAnalysis && (
            <motion.div
              key="loading"
              className="w-full bg-[#1F2937] p-8 rounded-xl shadow-lg border border-gray-700/50"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex flex-col items-center justify-center py-12">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="w-16 h-16 border-4 border-[#00C1D5] border-t-transparent rounded-full mb-4"
                />
                <h3 className="text-xl font-medium text-[#00C1D5]">
                  Preparing Visualizations...
                </h3>
                <p className="text-[#CBD5E1] mt-2">
                  This may take a moment as we process the training results (Attempt {pollingAttempts + 1}/{maxPollingAttempts}).
                  <br />
                  <strong>Debug Note:</strong> Check browser console (F12) for logs.
                </p>
                <Button
                  onClick={handleForceShowAnalysis}
                  className="mt-4 bg-red-500 hover:bg-red-600 text-white"
                >
                  Force Show Analysis (Debug)
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {showAnalysis && currentRunId && (
            <>
              <ErrorBoundary componentName="UMAPViewer">
                <motion.div
                  key="umap"
                  className="bg-[#1F2937] p-8 rounded-xl shadow-lg border border-gray-700/50"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <h2 className="text-2xl font-semibold mb-6 text-[#00C1D5]">Patient Embedding Space (UMAP)</h2>
                  <UMAPViewer runId={currentRunId} />
                </motion.div>
              </ErrorBoundary>

              <ErrorBoundary componentName="DivergenceViewer">
                <motion.div
                  key="divergence"
                  className="bg-[#1F2937] p-8 rounded-xl shadow-lg border border-gray-700/50"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5, delay: 0.4 }}
                >
                  <h2 className="text-2xl font-semibold mb-6 text-[#00C1D5]">Client Divergence Tracker</h2>
                  <DivergenceViewer runId={currentRunId} onLoadComplete={() => setShowImportance(true)} />
                </motion.div>
              </ErrorBoundary>

              <ErrorBoundary componentName="FeatureImportanceViewer">
                <motion.div
                  key="feature"
                  className="bg-[#1F2937] p-8 rounded-xl shadow-lg border border-gray-700/50"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <h2 className="text-2xl font-semibold mb-6 text-[#00C1D5]">Feature Importance Analysis</h2>
                  <FeatureImportanceViewer runId={currentRunId} onLoadComplete={() => setShowUMAP(true)} />
                </motion.div>
              </ErrorBoundary>
            </>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}