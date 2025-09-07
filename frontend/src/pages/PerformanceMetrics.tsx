"use client";
import Header from "@/components/Header";
import Footer from "@/components/Footer";   
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PolarAngleAxis, RadialBar, RadialBarChart } from "recharts";
import { ChartContainer } from "@/components/ui/chart";
import { useTrafficData } from "../components/trafficdataprovider";

// --- Reusable Radial Chart Component (No changes needed here) ---
const MetricRadialChart = ({ value, text, color }: { value: number; text: string; color: string }) => {
  const chartData = [{ name: 'metric', value, fill: color }];
  const chartColor = color || "#8884d8"; // Default color if not provided

  return (
    <div className="relative flex h-40 w-full items-center justify-center">
      <ChartContainer
        config={{ metric: { label: "Metric", color: chartColor } }}
        className="absolute inset-0"
      >
        <RadialBarChart data={chartData} startAngle={90} endAngle={-270} innerRadius={60} outerRadius={75} barSize={12} cy="50%">
          <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
          <RadialBar dataKey="value" cornerRadius={10} background={{ fill: "hsl(var(--muted))" }} />
        </RadialBarChart>
      </ChartContainer>
      <div className="flex flex-col items-center justify-center">
        <span className="font-hero text-3xl font-bold" style={{ color: chartColor }}>
          {text}
        </span>
      </div>
    </div>
  );
};

// --- Updated PerformanceMetrics Component ---
const PerformanceMetrics = () => {
  // Get live data from our context
  const { data, isConnected } = useTrafficData();
  const metrics = data?.performance_metrics;
  const mainDashboardData = data?.main_dashboard;

  const getStatusColor = (status: string) => {
    switch (status) {
      case "EXCELLENT": return "text-emerald-500 bg-emerald-500/10";
      case "GOOD": return "text-green-500 bg-green-500/10";
      case "AVERAGE": return "text-yellow-500 bg-yellow-500/10";
      case "POOR": return "text-red-500 bg-red-500/10";
      default: return "text-muted-foreground bg-muted";
    }
  };
  
  // Helper to extract the numerical value from a string like "78%"
  const parseValue = (valueStr: string = "0%") => {
      return parseInt(valueStr.replace('%', '').replace('s', '')) || 0;
  }

  // Handle loading and connection states
  if (!isConnected) {
    return <div className="flex h-screen items-center justify-center">Connecting to AI Traffic System...</div>;
  }
  
  if (!metrics || !mainDashboardData) {
    return <div className="flex h-screen items-center justify-center">Waiting for initial performance data...</div>;
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      {/* Main Content Area */}
      <main className="pt-24 pb-16">
        <div className="container mx-auto px-6">
          {/* Page Header */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} className="mb-12">
            <h1 className="font-hero text-5xl font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
              Performance Metrics
            </h1>
            <p className="font-body text-xl text-muted-foreground max-w-3xl">
              Live performance analysis based on AI output data.
            </p>
          </motion.div>

          {/* Core Metrics - Rendered from LIVE data */}
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.2 }} className="grid lg:grid-cols-3 gap-8 mb-12">
            {metrics.map((metric, index) => {
              const chartColor = metric.status === 'EXCELLENT' ? "#10b981" : metric.status === 'GOOD' ? "#22c55e" : "#f59e0b";
              return (
              <motion.div key={index} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.1 * index }}>
                <Card className="bg-card/50 backdrop-blur-sm border-border h-full flex flex-col">
                  <CardHeader className="pb-4">
                    <div className="flex items-start justify-between">
                      <CardTitle className="font-body text-lg font-semibold text-foreground leading-tight">
                        {metric.title}
                      </CardTitle>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(metric.status)}`}>
                        {metric.status}
                      </span>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0 flex flex-col flex-grow justify-center">
                    <MetricRadialChart
                      value={parseValue(metric.value)}
                      text={metric.value}
                      color={chartColor}
                    />
                    <div className="text-center font-body text-sm text-muted-foreground mt-4">
                      {metric.details}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )})}
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Card className="bg-card/50 backdrop-blur-sm border-border">
              <CardHeader>
                <CardTitle className="font-body text-xl font-semibold text-foreground">
                  Performance Summary
                </CardTitle>
                <p className="font-body text-muted-foreground">
                  Key insights from today's AI traffic management system performance
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <h3 className="font-body text-lg font-semibold text-foreground">
                      Decision Analytics
                    </h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="font-body text-muted-foreground">AI Decisions</span>
                        <span className="font-body font-medium">156</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-body text-muted-foreground">Emergency Overrides</span>
                        <span className="font-body font-medium">32</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-body text-muted-foreground">Queue Overrides</span>
                        <span className="font-body font-medium">12</span>
                      </div>
                      <div className="flex justify-between border-t pt-3">
                        <span className="font-body font-semibold">Total Decisions</span>
                        <span className="font-body font-semibold">200</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="font-body text-lg font-semibold text-foreground">
                      Traffic Impact
                    </h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="font-body text-muted-foreground">Current Avg Queue</span>
                        <span className="font-body font-medium">12.5 vehicles</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-body text-muted-foreground">Baseline (Pre-AI)</span>
                        <span className="font-body font-medium">20 vehicles</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-body text-muted-foreground">Emergency Responses</span>
                        <span className="font-body font-medium">8 today</span>
                      </div>
                      <div className="flex justify-between border-t pt-3">
                        <span className="font-body font-semibold text-emerald-600">Improvement</span>
                        <span className="font-body font-semibold text-emerald-600">37% reduction</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </main> 
      <Footer />  

    </div>
  );
};

export default PerformanceMetrics;
