export const WarningList = ({ warnings }: { warnings: string[] }) => {
  if (warnings.length === 0) return null;

  return (
    <div className="mt-4 p-3 bg-yellow-50 border-l-4 border-yellow-200 rounded-md">
      <h4 className="text-xl font-medium mb-2">Warnings:</h4>
      {warnings.map((warning, index) => (
        <div key={index} className="flex items-start space-x-2">
          <span className="text-red-500 text-lg">⚠️</span>
          <p className="text-sm text-gray-700">{warning}</p>
        </div>
      ))}
    </div>
  );
};