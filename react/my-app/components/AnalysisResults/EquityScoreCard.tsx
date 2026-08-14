import React from "react";

interface EquityScoreCardProps {
  equityScore: number;
  protectedAttributes: string[];
  targetColumns: string[];
}

export const EquityScoreCard = ({
  equityScore,
  protectedAttributes,
  targetColumns,
}: EquityScoreCardProps) => {
  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-4">
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <div>
          <p className="text-sm text-gray-500">Protected Attributes</p>
          <p className="font-medium text-lg">
            {protectedAttributes.join(", ") || "None"}
          </p>
        </div>
        <div>
          <p className="text-lg font-medium mb-1">Equity Score:</p>
          <p className="text-3xl font-bold">
            {equityScore?.toFixed(1) ?? "N/A"}
          </p>
        </div>
      </div>

      {/* Key Factors */}
      <div className="px-4 py-2">
        <p className="text-sm text-gray-600">Key factors:</p>
        {protectedAttributes.length > 0 ? (
          <p className="mt-1">
            • {protectedAttributes.join("<br/> • ")}
          </p>
        ) : (
          <p className="text-xs text-gray-500">None</p>
        )}
        {targetColumns.length > 0 ? (
          <p className="mt-2 text-sm text-gray-600">
            Targets: {targetColumns.join(" • ")}
          </p>
        ) : null}
      </div>
    </div>
  );
};