import { motion } from "framer-motion";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export const GlassPane = ({ children, className, onClick, ...props }) => {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", stiffness: 300, damping: 25 }}
      className={twMerge(
        "relative overflow-hidden rounded-3xl border border-white/10 bg-glass-100 backdrop-blur-xl shadow-2xl text-white",
        className
      )}
      onClick={onClick}
      {...props}
    >
      {/* Subtle Gradient Sheen */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />
      <div className="relative z-10 h-full w-full">{children}</div>
    </motion.div>
  );
};