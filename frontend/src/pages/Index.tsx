import React, { useState } from "react";
import Header from "@/components/Header";
import HeroText from "@/components/HeroText";
import SplineScene from "@/components/SplineScene";
import ScrollContent from "@/components/ScrollContent";
import Footer from "@/components/Footer";

const Index = () => {
  const [isLoading, setIsLoading] = useState(true);
  return (
    <div className="relative">
      (
        <>
          {/* Fixed 3D Background */}
          <SplineScene />
          
          {/* Header */}
          <Header />
          
          {/* Main Content */}
          <main className="relative z-10">
            {/* Hero Section with 3D Background */}
            <HeroText />
            
            {/* Scrollable Content */}
            <div className="bg-gradient-subtle">
              <ScrollContent />
            </div>
          </main>
          
          {/* Footer */}
          <Footer />
        </>
      )
    </div>
  );
};

export default Index;