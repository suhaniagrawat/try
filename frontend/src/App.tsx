import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "next-themes";
import Index from "./pages/Index";
import Dashboard from "./pages/Dashboard";
import PerformanceMetrics from "./pages/PerformanceMetrics";
import NotFound from "./pages/NotFound";
import EmergencyPage from "./pages/EmergencyPage";
// Corrected path assuming App.tsx is in src/
import { TrafficDataProvider } from "./components/trafficdataprovider";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      {/* Wrap the router with the TrafficDataProvider */}
      <TrafficDataProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/performance" element={<PerformanceMetrics />} />
              <Route path="/emergency" element={<EmergencyPage />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </TrafficDataProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;

