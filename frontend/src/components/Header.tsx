import { Button } from "@/components/ui/button";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { motion } from "framer-motion";

const Header = () => {
  const { theme, setTheme } = useTheme();

  return (
    <motion.header
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border"
    >
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <motion.h1 
            className="font-hero text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 17 }}
          >
            <a href="/">The Route Cause</a>
          </motion.h1>

          {/* Navigation */}
          <div className="flex items-center gap-4">
            <Button variant="ghost" className="font-body font-medium hover:bg-primary/10" asChild>
              <a href="/dashboard">Dashboard</a>
            </Button>
            <Button variant="ghost" className="font-body font-medium hover:bg-primary/10" asChild>
              <a href="/performance">Performance Metrics</a>
            </Button>
            <Button variant="ghost" className="font-body font-medium hover:bg-primary/10" asChild>
              <a href="/emergency">Emergency Mode</a>
            </Button>
            
            {/* Theme Toggle - Circle variant, top-right */}
            <Button
              variant="outline"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="rounded-full w-12 h-12 border-2 hover:bg-primary/10 hover:border-primary/30 transition-all duration-300"
            >
              <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;