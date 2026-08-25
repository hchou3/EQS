import React, { useState, forwardRef, useImperativeHandle } from "react";
import { EquityScoreCard } from "./EquityScoreCard";
import { DisparityTable } from "./DisparityTable";
import { WarningList } from "./WarningList";
import { LLMClassificationCard } from "./LLMClassificationCard";
import { CSVUploadResponse } from "@/lib/api/types";

export interface AnalysisResultsProps {
  analysisResult?: CSVUploadResponse;
}

type AnalysisImperativeMethods = {
  focus: () => void;
};

const AnalysisResults = forwardRef<
  AnalysisImperativeMethods,
  AnalysisResultsProps
>(({ analysisResult }, ref) => {
  const [show, setShow] = useState(false);

  useImperativeHandle(ref, () => ({
    focus: () => setShow(true),
  }));

  return (
    <div className="analysis-results overflow-auto">
      <div style={{ visibility: show ? "visible" : "hidden" }}>
        {analysisResult && (
          <>
            <EquityScoreCard
              equityScore={analysisResult.equity_score || 0}
              protectedAttributes={
                analysisResult.llm_classifications.protected_attributes
              }
              targetColumns={analysisResult.llm_classifications.target_columns}
            />

            <DisparityTable
              disparitySummary={analysisResult.disparity_summary}
              targetTypes={
                analysisResult.llm_classifications.target_column_types
              }
            />

            <LLMClassificationCard
              llmClassification={analysisResult.llm_classifications}
            />

            {analysisResult.warnings.length > 0 && (
              <WarningList warnings={analysisResult.warnings} />
            )}
          </>
        )}
      </div>
    </div>
  );
});

export default AnalysisResults;
