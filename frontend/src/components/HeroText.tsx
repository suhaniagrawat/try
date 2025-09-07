import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";

const HeroText = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"]
  });

  const y = useTransform(scrollYProgress, [0, 1], [0, -200]);
  const opacity = useTransform(scrollYProgress, [0, 0.7, 1], [1, 0.5, 0]);

  return (
    <div ref={containerRef} className="relative h-screen flex items-center justify-center">
      <motion.div
        style={{ y, opacity }}
        className="text-center z-10 px-6"
      >
        <motion.h1
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
          className="font-hero text-6xl md:text-8xl lg:text-9xl mb-6 bg-gradient-hero bg-clip-text text-transparent leading-tight"
        >
          The Algorithm 
        </motion.h1>
        
        <motion.h2
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 1, delay: 0.7, ease: "easeOut" }}
          className="font-hero text-6xl md:text-8xl lg:text-9xl mb-8 bg-gradient-hero bg-clip-text text-transparent leading-tight"
        >
          of the Open Road.
        </motion.h2>
      </motion.div>
    </div>
  );
};

export default HeroText;