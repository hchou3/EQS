import React from "react";

const CircularDotsLoader = () => {
  return (
    <div className="relative w-8 h-8 text-gray-300 bg-white rounded-full">
      <div className="absolute top-0 left-0 right-0 bottom-0 flex items-center justify-center opacity-75">
        {/* First Dot */}
        <span className="w-2 h-2 bg-[var(--outline)] rounded-full animate-pulse"></span>
        {/* Second Dot */}
        <span
          className="w-2 h-2 bg-[var(--outline)] rounded-full animate-pulse
                  transform: translateX(10px)"
        ></span>
        {/* Third Dot */}
        <span
          className="w-2 h-2 bg-[var(--outline)] rounded-full animate-pulse
                  transform: translateY(10px)"
        ></span>
        {/* Fourth Dot */}
        <span
          className="w-2 h-2 bg-[var(--outline)] rounded-full animate-pulse
                  transform: translateX(-10px) translateY(10px)"
        ></span>
      </div>
    </div>
  );
};
export default CircularDotsLoader;
