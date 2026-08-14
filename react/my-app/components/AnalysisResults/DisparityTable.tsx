import React, { useMemo } from "react";
import {
  useTable,
  tableFeatures,
  flexRender,
  type ColumnDef,
} from "@tanstack/react-table";
import { DisparitySummary } from "@/lib/api/types";

/**
 * Calculates CSS class names for table cells based on accessor type and value.
 */
const calculateCellClass = (
  type: string,
  accessor: string,
  value: any,
): string => {
  const classes: string[] = [];

  if (accessor === "disparity") {
    classes.push(
      type === "classification" ? "block" : "",
      value ? "text-center" : "text-gray-500",
    );
  }

  if (accessor === "p_value") {
    classes.push(value === null || value === "N/A" ? "text-gray-500" : "");
  }

  if (accessor === "status") {
    classes.push(
      value === "Significant" ? "text-green-800 font-medium" : "text-gray-600",
    );
  }

  return classes.filter(Boolean).join(" ");
};

export interface DisparityTableProps {
  disparitySummary: DisparitySummary[];
  targetTypes: Record<string, string>;
}

type DisparityRow = DisparitySummary & { status: string };

// No sorting/filtering/pagination is actually wired up in this table,
// so we don't need to register any extra features.
const features = tableFeatures({});

const columnDefs: ColumnDef<typeof features, DisparityRow>[] = [
  { header: "Protected Attr", accessorKey: "protected_attr" },
  { header: "Target Column", accessorKey: "target_col" },
  { header: "Task Type", accessorKey: "task_type" },
  { header: "Disparity", accessorKey: "disparity" },
  { header: "P-value", accessorKey: "p_value" },
  { header: "Status", accessorKey: "status" },
];

export const DisparityTable = ({
  disparitySummary,
  targetTypes,
}: DisparityTableProps) => {
  // Map the data to add the status field (memoized: v9 wants stable refs)
  const data = useMemo<DisparityRow[]>(
    () =>
      disparitySummary.map((item) => ({
        ...item,
        status:
          item.p_value !== null && item.p_value < 0.05
            ? "Significant"
            : item.p_value !== null
              ? "Not Significant"
              : "N/A",
      })),
    [disparitySummary],
  );

  const table = useTable({
    features,
    columns: columnDefs,
    data,
  });

  // Early return for empty data
  if (disparitySummary.length === 0) {
    return (
      <p className="text-center text-gray-500 py-4">
        No disparity data available
      </p>
    );
  }

  return (
    <div className="mt-4 p-4 bg-white rounded-lg shadow w-full">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext(),
                      )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id}>
              {row.getAllCells().map((cell) => (
                <td
                  key={cell.id}
                  className={`px-6 py-4 whitespace-nowrap text-sm ${calculateCellClass(
                    row.original.task_type,
                    cell.column.id,
                    cell.getValue(),
                  )}`}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination Controls */}
      <div className="mt-4 flex items-center justify-between px-6 py-3 bg-gray-50 rounded-lg">
        <button
          onClick={() => {
            // Implementation would go here
          }}
          disabled
          className="px-4 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          Previous
        </button>

        <span className="text-sm text-gray-700">
          Page <span className="font-medium">1</span> of{" "}
          <span className="font-medium">1</span>
        </span>

        <button
          onClick={() => {
            // Implementation would go here
          }}
          disabled
          className="px-4 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
};
