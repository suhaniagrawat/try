import React, { useState, useEffect } from 'react';

const LoadingScreen = ({ setLoadingComplete }) => {
  const [progress, setProgress] = useState(0);
  const [loadingText, setLoadingText] = useState('Initializing Systems...');
  
  // Array of messages to display during loading
  const messages = [
    'Initializing Systems...',
    'Loading 3D Assets...',
    'Configuring UI Elements...',
    'Finalizing Experience...',
    'Welcome!',
  ];
  
  // Use useEffect to manage the loading process
  useEffect(() => {
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += 1;
      // Simulate loading for 12 seconds
      const targetProgress = Math.min(Math.floor((currentProgress / 1200) * 100), 100);
      setProgress(targetProgress);

      if (currentProgress > 1000) {
        setLoadingText(messages[4]);
      } else if (currentProgress > 750) {
        setLoadingText(messages[3]);
      } else if (currentProgress > 500) {
        setLoadingText(messages[2]);
      } else if (currentProgress > 250) {
        setLoadingText(messages[1]);
      } else {
        setLoadingText(messages[0]);
      }

      if (currentProgress >= 1200) {
        clearInterval(interval);
        // Corrected: Set the loading state to false
        setTimeout(() => setLoadingComplete(false), 500); 
      }
    }, 10); // Update every 10ms for smooth animation

    return () => clearInterval(interval);
  }, [setLoadingComplete]);

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center p-4 text-white transition-opacity duration-500 bg-dark-background opacity-100">
      <div className="max-w-2xl w-full text-center">
        <h1 className="text-4xl md:text-5xl font-jaro text-primary-purple transition-all duration-300">
          Loading
        </h1>
        {/* White boundary with a purple loading bar */}
        <div className="mt-8 relative h-2 w-full rounded-full border border-white">
          <div 
            className="absolute inset-y-0 left-0 bg-primary-purple rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <div className="flex items-center justify-between mt-4">
          <p className="text-sm md:text-base font-inter text-supporting-text">{loadingText}</p>
          <span className="text-xl md:text-2xl font-jaro text-primary-purple transition-transform transform duration-300">{progress}%</span>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;