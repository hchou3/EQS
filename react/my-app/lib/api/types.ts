export interface CSVUploadResponse {
  session_id: string;
  filename: string;
  dataset_info: DatasetInfoSummary;
  llm_classifications: LLMClassification;
  equity_score: number | null;
  disparity_summary: DisparitySummary[];
  warnings: string[];
}

export interface DatasetInfoSummary {
  n_rows: number;
  n_cols: number;
  column_names: string[];
}

export interface ColumnReasoning {
  protected_attributes_explanation: string;
  target_columns_explanation: string;
  domain_assessment: string;
}

export interface ColumnReasoningDetails {
  rationale: string;
  confidence: string;
}

export type FavorableDirection = "higher" | "lower" | "neutral";

export interface FavorableDirectionInfo {
  direction: FavorableDirection;
  rationale: string;
  over_predict_consequence: string;
  under_predict_consequence: string;
  confidence: string;
}

export interface DisparitySummary {
  protected_attr: string;
  target_col: string;
  task_type: string; // "classification" | "regression"
  disparity: number | null;
  p_value: number | null;
  error: string | null;
}

export interface LLMClassification {
  protected_attributes: string[];
  target_columns: string[];
  target_column_types: Record<string, string>;
  reasoning: ColumnReasoning;
  regression_favorable_directions: Record<string, FavorableDirectionInfo>;
}
