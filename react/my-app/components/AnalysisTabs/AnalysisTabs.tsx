import React, { useState } from "react";
import { EquityScoreCard } from "../AnalysisResults/EquityScoreCard";
import { DisparityTable } from "../AnalysisResults/DisparityTable";
import { LLMClassificationCard } from "../AnalysisResults/LLMClassificationCard";
import { WarningList } from "../AnalysisResults/WarningList";
import { CSVUploadResponse } from "@/lib/api/types";

interface AnalysisTabsProps {
  analysisResult?: CSVUploadResponse;
}

export default function AnalysisTabs({ analysisResult }: AnalysisTabsProps) {
  const [activeTab, setActiveTab] = useState("equity");

  const tabs = [
    {
      id: "equity",
      title: "Equity Score",
      icon: "📊",
      content: (
        <EquityScoreCard
          equityScore={analysisResult?.equity_score || 0}
          protectedAttributes={
            analysisResult?.llm_classifications.protected_attributes || []
          }
          targetColumns={
            analysisResult?.llm_classifications.target_columns || []
          }
        />
      ),
    },
    {
      id: "disparities",
      title: "Disparities",
      icon: "📈",
      content: analysisResult?.disparity_summary &&
      analysisResult.disparity_summary.length > 0 ? (
        <DisparityTable
          disparitySummary={analysisResult.disparity_summary}
          targetTypes={
            analysisResult.llm_classifications.target_column_types || {}
          }
        />
      ) : (
        <p className="text-center text-gray-500 py-4">
          No disparity data available
        </p>
      ),
    },
    {
      id: "llm-classification",
      title: "LLM Classification",
      icon: "🤖",
      content: analysisResult?.llm_classifications ? (
        <LLMClassificationCard
          llmClassification={analysisResult.llm_classifications}
        />
      ) : (
        <p className="text-center text-gray-500 py-4">
          No classification data available
        </p>
      ),
    },
    {
      id: "warnings",
      title: "Warnings",
      icon: "⚠️",
      content:
        analysisResult?.warnings && analysisResult.warnings.length > 0 ? (
          <WarningList warnings={analysisResult.warnings} />
        ) : (
          <p className="text-center text-gray-500 py-4">
            No warnings to display
          </p>
        ),
    },
  ];

  return (
    <div className="bg-[var(--surface)] rounded-lg shadow-lg p-4">
      {/* Tab Bar */}
      <div className="flex space-x-2 mb-4 border-b border-[var(--outline-muted)]">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-t-lg
              transition-colors
              ${
                activeTab === tab.id
                  ? "bg-[var(--surface-elevated)] text-[var(--outline)] border-b-2 border-[var(--outline)]"
                  : "text-[var(--foreground)]/70 hover:text-[var(--outline)] hover:bg-[var(--surface-elevated)/30]"
              }
            `}
          >
            <span>{tab.icon}</span>
            {tab.title}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="mt-4">
        {tabs.map(
          (tab) =>
            activeTab === tab.id && (
              <div key={tab.id} className="animate-fadeIn">
                {tab.content}
              </div>
            ),
        )}
      </div>
    </div>
  );
}
