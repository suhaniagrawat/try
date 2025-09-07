"use client";
import Header from "@/components/Header";
import Footer from "@/components/Footer"; 
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { Progress } from "@/components/ui/progress";
import React, { useState, useEffect } from "react";
// --- 1. IMPORT THE CUSTOM HOOK ---
// Corrected path to be a relative import
import { useTrafficData } from "../components/trafficdataprovider";

// Mock components since we don't have the files


// This constant is used to calculate the queue percentage.
// It should match the MAX_VEHICLES_PER_LANE in your Python script.
const MAX_VEHICLES_PER_LANE = 40;

const Dashboard = () => {
  // --- 2. GET LIVE DATA FROM THE CONTEXT ---
  const { data, isConnected } = useTrafficData();

  // Local state for the chart data, which will accumulate over time
  const [queueTrends, setQueueTrends] = useState<any[]>([]);

  // --- 3. EFFECT TO UPDATE THE CHART DATA ---
  useEffect(() => {
    if (data?.main_dashboard?.vehicle_counters) {
      const now = new Date();
      // Format time as HH:MM for the chart's X-axis
      const timeLabel = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
      
      const newTrendPoint = {
        time: timeLabel,
        ...data.main_dashboard.vehicle_counters,
      };

      setQueueTrends(prevTrends => {
        // Keep only the last 20 data points for a clean and moving chart
        const updatedTrends = [...prevTrends, newTrendPoint];
        return updatedTrends.slice(-20); 
      });
    }
  }, [data]); // This effect runs every time 'data' from the context changes


  // --- Helper to render a loading/disconnected state ---
  if (!isConnected || !data) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
          <div className="text-center">
              <h1 className="font-hero text-2xl font-bold mb-4">
                  {isConnected ? "Waiting for first update from AI..." : "Connecting to Traffic System..."}
              </h1>
              <p className="text-muted-foreground">Please ensure the backend server and AI agent are running.</p>
          </div>
      </div>
    );
  }

  // --- Destructure live data for easier access in the component ---
  const { main_dashboard } = data;
  const signal_state = main_dashboard?.signal_state || {};
  const vehicleCounts = main_dashboard?.vehicle_counters || {};
  const activeDirection = signal_state.active_direction || 'None';
  const stateColor = signal_state.state || 'RED';
  
  // Get the lane names dynamically from the data for robust rendering
  const laneNames = Object.keys(vehicleCounts);

  // Reusable Signal Light component that now reads from live data
  const SignalLight = ({ direction }: { direction: string }) => {
    const isGreen = activeDirection === direction && stateColor === 'GREEN';
    const isYellow = activeDirection === direction && stateColor === 'YELLOW';
    const isRed = !isGreen && !isYellow;

    const color = isGreen ? 'GREEN' : isYellow ? 'YELLOW' : 'RED';
    const timer = isGreen || isYellow ? signal_state.timer || 0 : '-';

    return (
      <div className="flex flex-col items-center space-y-2">
        <div className="bg-muted rounded-lg p-3 flex flex-col items-center space-y-2">
          <div className={`w-10 h-10 rounded-full border-2 border-border flex items-center justify-center transition-colors ${isRed ? 'bg-red-500/50' : ''}`}>
            {isRed && <div className="w-6 h-6 rounded-full bg-red-500"></div>}
          </div>
          <div className={`w-10 h-10 rounded-full border-2 border-border flex items-center justify-center transition-colors ${isYellow ? 'bg-yellow-500/50' : ''}`}>
             {isYellow && <div className="w-6 h-6 rounded-full bg-yellow-500"></div>}
          </div>
          <div className={`w-10 h-10 rounded-full border-2 border-border flex items-center justify-center transition-colors ${isGreen ? 'bg-green-500/50' : ''}`}>
            {isGreen && <div className="w-6 h-6 rounded-full bg-green-500"></div>}
          </div>
        </div>
        <div className="text-center">
          <div className="font-body text-xs text-muted-foreground">{direction}</div>
          <div className="font-hero text-lg font-bold text-foreground">{timer}{typeof timer === 'number' ? 's' : ''}</div>
          <div className="font-body text-xs text-muted-foreground">{color}</div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-6">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} className="mb-12">
            <h1 className="font-hero text-5xl font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
              Traffic Control Dashboard
            </h1>
            <p className="font-body text-xl text-muted-foreground max-w-3xl">
              Real-time intersection monitoring with signal control, vehicle counting, and queue management.
            </p>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.2 }} className="grid lg:grid-cols-2 gap-8 mb-12">
            <Card className="bg-card/50 backdrop-blur-sm border-border">
              <CardHeader>
                <CardTitle className="font-body text-xl font-semibold text-foreground">
                  Traffic Signal Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  {laneNames.map((direction) => (
                    <div key={direction} className="flex justify-center">
                      <SignalLight direction={direction} />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card/50 backdrop-blur-sm border-border">
              <CardHeader>
                <CardTitle className="font-body text-xl font-semibold text-foreground">
                  Backend System Live Feed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted/30 rounded-lg border-2 border-dashed border-border overflow-hidden">
                  <video
                    src="../video/feed.mp4" // Path to your video in the `public` folder
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="w-full h-full object-cover" // `object-cover` ensures the video fills the container without distortion
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.3 }} className="mb-12">
            <Card className="bg-card/50 backdrop-blur-sm border-border">
              <CardHeader>
                <CardTitle className="font-body text-xl font-semibold text-foreground">
                  Live Vehicle Counters
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(vehicleCounts).map(([direction, count]) => (
                    <div key={direction} className="text-center p-4 bg-muted/50 rounded-lg">
                      <div className="font-body text-sm text-muted-foreground mb-1">{direction}</div>
                      <div className="font-hero text-2xl font-bold text-foreground">{count}</div>
                      <div className="font-body text-xs text-muted-foreground">vehicles</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.4 }} className="mb-12">
            <Card className="bg-card/50 backdrop-blur-sm border-border">
              <CardHeader>
                <CardTitle className="font-body text-xl font-semibold text-foreground">
                  Queue Length Indicators
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {Object.entries(vehicleCounts).map(([direction, count]) => {
                    const queueLength = Math.min(100, (count / MAX_VEHICLES_PER_LANE) * 100);
                    return (
                      <div key={direction} className="space-y-2">
                        <div className="flex justify-between items-center">
                          <div className="font-body text-sm font-medium text-foreground">{direction}</div>
                          <div className="font-hero text-lg font-bold text-foreground">{Math.round(queueLength)}%</div>
                        </div>
                        <Progress 
                          value={queueLength} 
                          className="h-3 bg-muted/50" 
                        />
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.6 }}>
            <Card className="bg-card/50 backdrop-blur-sm border-border">
              <CardHeader>
                <CardTitle className="font-body text-xl font-semibold text-foreground">
                  Queue Trends Over Time
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ChartContainer config={{}} className="h-[600px]">
                  <ResponsiveContainer>
                    <AreaChart data={queueTrends}>
                      <defs>
                        <linearGradient id="fillNorthbound" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} /><stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.1} /></linearGradient>
                        <linearGradient id="fillSouthbound" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="hsl(var(--primary-glow))" stopOpacity={0.3} /><stop offset="95%" stopColor="hsl(var(--primary-glow))" stopOpacity={0.1} /></linearGradient>
                        <linearGradient id="fillEastbound" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="hsl(var(--accent))" stopOpacity={0.3} /><stop offset="95%" stopColor="hsl(var(--accent))" stopOpacity={0.1} /></linearGradient>
                        <linearGradient id="fillWestbound" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="hsl(var(--secondary))" stopOpacity={0.1} /><stop offset="95%" stopColor="hsl(var(--secondary))" stopOpacity={0.0} /></linearGradient>
                      </defs>
                      <XAxis dataKey="time" />
                      <YAxis />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Area type="monotone" dataKey="Northbound" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#fillNorthbound)" />
                      <Area type="monotone" dataKey="Southbound" stroke="hsl(var(--primary-glow))" fillOpacity={1} fill="url(#fillSouthbound)" />
                      <Area type="monotone" dataKey="Eastbound" stroke="hsl(var(--accent))" fillOpacity={1} fill="url(#fillEastbound)" />
                      <Area type="monotone" dataKey="Westbound" stroke="hsl(var(--secondary))" fillOpacity={1} fill="url(#fillWestbound)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Dashboard;

