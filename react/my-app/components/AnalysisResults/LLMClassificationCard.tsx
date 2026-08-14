import { LLMClassification } from "@/lib/api/types";

interface LLMClassificationCardProps {
  llmClassification: LLMClassification;
}

export const LLMClassificationCard = ({
  llmClassification,
}: LLMClassificationCardProps) => {
  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-4">
      <div className="mb-4">
        <h4 className="text-lg font-medium mb-2">Model Analysis</h4>
        <p className="text-sm">Column classifications and reasoning</p>
      </div>

      <div className="flex flex-col">
        {llmClassification.protected_attributes.length > 0 && (
          <div className="flex flex-col md:flex-row gap-6 space-x-4 md:space-x-8 md:space-y-4">
            {llmClassification.protected_attributes.map((attr) => (
              <div key={attr} className="flex-1 overflow-x-auto space-x-4">
                {attr}
              </div>
            ))}
          </div>
        )}

        {llmClassification.reasoning &&
          llmClassification.reasoning.protected_attributes_explanation && (
            <div className="mt-4 text-sm text-gray-600">
              <p>Protected Attributes Reasoning</p>
              <div className="mt-2 text-right text-sm italic">
                {llmClassification.reasoning.protected_attributes_explanation}
              </div>
            </div>
          )}

        {llmClassification.regression_favorable_directions &&
          Object.keys(llmClassification.regression_favorable_directions)
            .length > 0 && (
            <div className="mt-4">
              <h5 className="text-sm font-medium mb-2">Regression Insights</h5>
              {Object.entries(
                llmClassification.regression_favorable_directions,
              ).map(([col, info]) => (
                <div key={col}>
                  <div className="mb-3">
                    <span className="text-xs font-bold">Direction:</span>{" "}
                    {info.direction}
                    <span className="ml-8 text-xs text-gray-500">
                      Confidence: {info.confidence}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
      </div>
    </div>
  );
};
