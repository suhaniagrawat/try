"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// --- Define the shape of our data ---
// This should exactly match the JSON payload from the Python backend
interface MainDashboardData {
  signal_state: { [key: string]: any };
  vehicle_counters: { [key: string]: number };
  total_vehicles: number;
}

interface PerformanceMetric {
  title: string;
  value: string;
  status: string;
  details: string;
}

interface EmergencyModeData {
  priority_direction: string;
  delayed_vehicles: number;
  total_vehicles: number;
}

export interface TrafficData {
  main_dashboard: MainDashboardData | null;
  performance_metrics: PerformanceMetric[];
  emergency_mode: EmergencyModeData | null;
}

// Define the shape of our Context
interface TrafficDataContextType {
  data: TrafficData | null;
  isConnected: boolean;
}

// --- Create the Context ---
const TrafficDataContext = createContext<TrafficDataContextType | undefined>(undefined);

// Define the WebSocket URL
//const WEBSOCKET_URL = "ws://127.0.0.1:8000/ws/dashboard";
const WEBSOCKET_URL = process.env.VITE_WEBSOCKET_URL || "ws://localhost:8000/ws/dashboard";
// --- Create the Provider Component ---
// This component will wrap your entire application. It handles the actual
// connection and provides the data to all children.
export const TrafficDataProvider = ({ children }: { children: ReactNode }) => {
  const [data, setData] = useState<TrafficData | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const socket = new WebSocket(WEBSOCKET_URL);

    socket.onopen = () => {
      console.log("✅ WebSocket connection established.");
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const receivedData: TrafficData = JSON.parse(event.data);
        setData(receivedData);
      } catch (error) {
        console.error("❌ Error parsing WebSocket message:", error);
      }
    };

    socket.onclose = () => {
      console.log("🔌 WebSocket connection closed.");
      setIsConnected(false);
      // In a production app, you might want a more robust reconnection strategy
    };

    socket.onerror = (error) => {
      console.error("❌ WebSocket error:", error);
      setIsConnected(false);
    };

    // Cleanup function: close the connection when the component is unmounted
    return () => {
      socket.close();
    };
  }, []); // The empty dependency array ensures this effect runs only once

  const value = { data, isConnected };

  return (
    <TrafficDataContext.Provider value={value}>
      {children}
    </TrafficDataContext.Provider>
  );
};

// --- Create a Custom Hook ---
// This makes it easy for any component to get the live data.
export const useTrafficData = () => {
  const context = useContext(TrafficDataContext);
  if (context === undefined) {
    throw new Error('useTrafficData must be used within a TrafficDataProvider');
  }
  return context;
};
