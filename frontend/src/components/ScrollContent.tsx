import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import { Button } from "@/components/ui/button";
import { ArrowRight, BarChart3, Target, TrendingUp } from "lucide-react";

const ScrollContent = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  const y = useTransform(scrollYProgress, [0, 1], [100, -100]);

  const features = [
    {
      icon: <BarChart3 className="h-8 w-8" />,
      title: "Live Traffic Analysis ",
      description: "Our state-of-the-art computer vision (YOLOv8) sees every vehicle and pedestrian, providing a complete, real-time understanding of intersection dynamics."
    },
    {
      icon: <Target className="h-8 w-8" />,
      title: "Adaptive AI Control",
      description: "A trained Reinforcement Learning agent analyzes live data to make strategic decisions, dynamically optimizing signal timings to eliminate bottlenecks and reduce wait times.      "
    },
    {
      icon: <TrendingUp className="h-8 w-8" />,
      title: "Priority & Safety Engine",
      description: "Our system ensures safety and fairness with a supervisory engine that provides instant emergency vehicle preemption and guarantees no driver is left waiting indefinitely."
    }
  ];

  return (
    <div ref={containerRef} className="relative min-h-screen py-20">
      <motion.div 
        style={{ y }}
        className="container mx-auto px-6"
      >
        {/* Section 1: Value Proposition */}
        <motion.section
          initial={{ opacity: 0, y: 60 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          viewport={{ once: true }}
          className="text-center mb-32"
        >
          <h2 className="font-hero text-5xl md:text-7xl mb-8 bg-gradient-primary bg-clip-text text-transparent">
            We've All Been There
          </h2>
          <p className="font-body text-xl md:text-2xl text-muted-foreground max-w-4xl mx-auto leading-relaxed mb-12">
            You know the feeling: you're stuck at a red light at 3 a.m. on a completely empty road. Traditional traffic lights are like that friend who can't read the room—they just stick to the plan, no matter how ridiculous it becomes.

            Our system is the friend who actually looks around and says, "Hey, no one's here. Let's get you home already."
          </p>
          <Button className="bg-gradient-primary hover:shadow-glow text-primary-foreground px-8 py-4 text-lg font-medium group">
            <a href="/dashboard" className="flex items-center">
              Get Started
            </a>
            <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
          </Button>
        </motion.section>

        {/* Section 2: Features */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 1, ease: "easeOut" }}
          viewport={{ once: true }}
          className="mb-32"
        >
          <div className="grid md:grid-cols-3 gap-12">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 40 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2, ease: "easeOut" }}
                viewport={{ once: true }}
                className="text-center group"
              >
                <div className="bg-gradient-primary w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6 text-primary-foreground group-hover:shadow-glow transition-all duration-300">
                  {feature.icon}
                </div>
                <h3 className="font-body text-2xl font-bold mb-4 text-foreground">
                  {feature.title}
                </h3>
                <p className="font-body text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Section 3: Call to Action */}
        <motion.section
          initial={{ opacity: 0, y: 60 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          viewport={{ once: true }}
          className="text-center"
        >
          <h2 className="font-hero text-4xl md:text-6xl mb-6 text-foreground">
           If time is money, 


          </h2>
          <h2 className="font-hero text-4xl md:text-6xl mb-8 bg-gradient-primary bg-clip-text text-transparent">
          isn't a fixed-timer traffic grid just organized,
          </h2>
          <h2 className="font-hero text-4xl md:text-6xl mb-8 bg-gradient-primary bg-clip-text text-transparent">
           city-wide theft?
          </h2>
          <p className="font-body text-xl text-muted-foreground mb-12 max-w-2xl mx-auto">
           Our AI traffic system is designed to stop this theft. By replacing blind, fixed timers with real-time perception and intelligent decision-making, we treat time as the valuable asset it is. Our system ensures every second of a green light is allocated efficiently based on actual traffic demand, returning that stolen value directly to a city's drivers, businesses, and its overall economy.








          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button className="bg-gradient-primary hover:shadow-glow text-primary-foreground px-8 py-4 text-lg font-medium">
              <a href="/Performance">performence Metric</a>
            </Button>
            <Button variant="outline" className="px-8 py-4 text-lg font-medium border-primary text-primary hover:bg-primary/10">
              <a href="/emergency">Emergency Mode</a>
            </Button>
          </div>
        </motion.section>
      </motion.div>
    </div>
  );
};

export default ScrollContent;