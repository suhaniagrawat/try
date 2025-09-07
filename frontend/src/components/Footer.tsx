import { motion } from "framer-motion";
import { Github, Twitter, Linkedin, Mail } from "lucide-react";
import { Button } from "@/components/ui/button";

const Footer = () => {
  const socialLinks = [
    { icon: <Twitter className="h-5 w-5" />, href: "#", label: "Twitter" },
    { icon: <Linkedin className="h-5 w-5" />, href: "#", label: "LinkedIn" },
    { icon: <Github className="h-5 w-5" />, href: "#", label: "GitHub" },
    { icon: <Mail className="h-5 w-5" />, href: "#", label: "Email" },
  ];

  return (
    <footer className="relative bg-gradient-subtle border-t border-border">
      <div className="container mx-auto px-6 py-16">
        <div className="grid md:grid-cols-2 gap-12 items-start">
          {/* About Section */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            viewport={{ once: true }}
          >
            <h3 className="font-hero text-3xl font-bold mb-6 bg-gradient-primary bg-clip-text text-transparent">
              The Route Cause
            </h3>
            <p className="font-body text-muted-foreground leading-relaxed mb-6 max-w-md">
            At The Route Cause, we use a multi-layered AI to solve traffic congestion. Our platform uses computer vision to see and a Reinforcement Learning brain to intelligently manage live traffic, all governed by a safety and optimization engine. The result is safer streets, faster commutes, quicker emergency response, and a cleaner environment.
            </p>
            <p className="font-body text-sm text-muted-foreground">
              
Join us in creating smarter, more responsive cities for everyone.
            </p>
          </motion.div>

          {/* Social Links */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            viewport={{ once: true }}
            className="text-center md:text-right"
          >
            <h4 className="font-body text-xl font-semibold mb-6 text-foreground">
              Connect With Us
            </h4>
            <div className="flex flex-wrap justify-center md:justify-end gap-4 mb-8">
              {socialLinks.map((link, index) => (
                <motion.div
                  key={index}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    className="bg-muted hover:bg-primary hover:text-primary-foreground transition-all duration-300"
                    asChild
                  >
                    <a href="https://www.linkedin.com/in/suhani-deepak-agrawat" aria-label={link.label}>
                      {link.icon}
                    </a>
                  </Button>
                </motion.div>
              ))}
            </div>
            <p className="font-body text-sm text-muted-foreground">
              ©️ 2024 The Route Cause. All rights reserved.
            </p>
          </motion.div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;