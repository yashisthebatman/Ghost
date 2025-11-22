import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Pause, Trophy, Activity, Zap, ChevronRight, Timer } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import { motion } from "framer-motion";
import TrackMap from './components/TrackMap';
import { useSimulationStore } from './store/simulationStore';
import './App.css';

const API_URL = "http://localhost:8000";

function App() {
  const [state, setState] = useState({ status: "BOOT", session: null, laps: [] });
  const [activeLapId, setActiveLapId] = useState(3);
  const [displayData, setDisplayData] = useState([]);
  const [ghostData, setGhostData] = useState([]);

  const { isPlaying, currentIndex, actions } = useSimulationStore();
  const requestRef = useRef();

  // --- BOOT ---
  useEffect(() => {
    const boot = async () => {
      try {
        await axios.get(`${API_URL}/`);
        const [ctx, laps, ghost] = await Promise.all([
            axios.get(`${API_URL}/session/context`),
            axios.get(`${API_URL}/session/laps`),
            axios.get(`${API_URL}/laps/ghost`),
        ]);

        setState({ status: "READY", session: ctx.data, laps: laps.data });
        setGhostData(ghost.data);
        
        // Load Lap 3 (PB) initially
        loadLapData(3, ghost.data);

      } catch (e) {
        console.error(e);
        setState(s => ({ ...s, status: "ERROR" }));
      }
    };
    boot();
  }, []);

  const loadLapData = async (lapId, ghostBase) => {
    setActiveLapId(lapId);
    actions.reset();
    try {
        const res = await axios.get(`${API_URL}/laps/actual/${lapId}`);
        const realData = res.data;
        const maxLength = Math.max(ghostBase.length, realData.length);
        
        const merged = Array.from({ length: maxLength }).map((_, i) => ({
            time: i * 0.01,
            ghost_speed: ghostBase[i]?.speed || 0,
            real_speed: realData[i]?.speed || 0,
            ghost_throttle: ghostBase[i]?.ath || 0,
            ghost_brake: ghostBase[i]?.pbrake_f || 0,
            ghost_steer: ghostBase[i]?.Steering_Angle || 0,
        }));

        setDisplayData(merged);
        actions.setDataLength(merged.length);
    } catch (e) { console.error(e); }
  };

  // --- ANIMATION ---
  const animate = () => {
    if (useSimulationStore.getState().isPlaying) {
      actions.nextFrame();
      requestRef.current = requestAnimationFrame(animate);
    }
  };
  useEffect(() => {
    if (isPlaying) requestRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(requestRef.current);
  }, [isPlaying]);

  // Helpers
  const frame = displayData[currentIndex] || {};
  const progress = displayData.length > 0 ? currentIndex / displayData.length : 0;
  const delta = (frame.ghost_speed - frame.real_speed) || 0;

  if (state.status === "BOOT") return <div className="h-screen w-screen bg-black flex items-center justify-center text-[10px] font-mono tracking-[0.3em] text-white/40 animate-pulse">SYSTEM STARTUP</div>;

  return (
    // STRICT GRID LAYOUT: No Flexbox Growing issues
    <div className="h-screen w-screen grid grid-rows-[48px_1fr] bg-black text-white font-sans overflow-hidden">
      
      {/* 1. HEADER (Fixed 48px) */}
      <header className="glass border-b border-white/5 flex items-center justify-between px-4 z-50">
        <div className="flex items-center gap-3">
            <div className="w-6 h-6 bg-red-600 rounded flex items-center justify-center text-[10px] font-bold shadow-lg">G</div>
            <div className="flex flex-col leading-none">
                <span className="text-xs font-bold tracking-wide">GHOST ENGINEER</span>
                <span className="text-[9px] text-white/40 font-mono">{state.session.vehicle_id}</span>
            </div>
        </div>
        <div className="flex gap-3">
            <Badge icon={<Zap size={10}/>} label={`${state.session.weather.track_temp}°C`} />
            <Badge icon={<Activity size={10} className="text-green-400"/>} label="ONLINE" />
        </div>
      </header>

      {/* 2. MAIN CONTENT (Grid Columns) */}
      <div className="grid grid-cols-[240px_1fr] gap-0 min-h-0">
        
        {/* SIDEBAR (Fixed 240px) */}
        <div className="glass border-r border-white/5 flex flex-col z-40">
            <div className="p-4 border-b border-white/5">
                <div className="text-[9px] font-bold text-white/30 tracking-widest uppercase mb-1">Session Laps</div>
                <div className="text-lg font-bold">{state.laps.length} Recorded</div>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {state.laps.map(lap => (
                    <div key={lap.lap_number} 
                         onClick={() => loadLapData(lap.lap_number, ghostData)}
                         className={`p-2 rounded-lg cursor-pointer transition-all border ${activeLapId === lap.lap_number ? 'bg-white/10 border-white/10' : 'border-transparent hover:bg-white/5 opacity-60 hover:opacity-100'}`}>
                        <div className="flex justify-between items-center">
                            <span className="text-[10px] font-bold">LAP {lap.lap_number}</span>
                            {lap.status === "PB" && <Trophy size={10} className="text-yellow-500"/>}
                        </div>
                        <div className="flex justify-between items-baseline mt-1">
                             <span className="text-sm font-mono font-medium">{lap.lap_time.toFixed(3)}s</span>
                             {activeLapId === lap.lap_number && <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"/>}
                        </div>
                    </div>
                ))}
            </div>
        </div>

        {/* CENTER STAGE (Grid Rows for Map/Telemetry) */}
        <div className="grid grid-rows-[1fr_200px] min-h-0 bg-gradient-to-b from-black/50 to-black/80 relative">
            
            {/* MAP AREA (Top) */}
            <div className="relative flex items-center justify-center p-4 overflow-hidden">
                <div className="w-full h-full max-w-[90%] max-h-[90%]">
                    <TrackMap className="w-full h-full" progress={progress} ghostProgress={progress} />
                </div>

                {/* HUD */}
                <div className="absolute top-4 left-4 glass px-3 py-1.5 rounded-lg flex items-center gap-3">
                    <Timer size={12} className="text-white/50"/>
                    <span className="text-lg font-mono font-bold">{(frame.time || 0).toFixed(2)}<span className="text-xs text-white/40">s</span></span>
                </div>

                <div className="absolute top-4 right-4 text-right">
                    <div className="text-3xl font-mono font-bold tracking-tighter">{Math.round(frame.ghost_speed || 0)}</div>
                    <div className="text-[9px] font-bold text-white/30 -mt-1 tracking-widest">KPH</div>
                    <div className={`text-xs font-mono font-bold mt-1 ${delta > 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {delta > 0 ? '+' : ''}{delta.toFixed(1)}
                    </div>
                </div>
            </div>

            {/* TELEMETRY DECK (Bottom 200px) */}
            <div className="glass border-t border-white/5 p-4 flex flex-col gap-4 z-30">
                {/* Chart */}
                <div className="flex-1 relative w-full">
                    <div className="absolute top-0 left-2 flex gap-3 text-[9px] font-bold tracking-wider z-10">
                        <span className="text-red-500 flex items-center gap-1"><div className="w-1 h-1 rounded-full bg-red-500"/> GHOST</span>
                        <span className="text-white/50 flex items-center gap-1"><div className="w-1 h-1 rounded-full bg-white/50"/> REAL</span>
                    </div>
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={displayData.slice(Math.max(0, currentIndex-100), Math.min(displayData.length, currentIndex+100))}>
                            <defs><linearGradient id="grad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ef4444" stopOpacity={0.2}/><stop offset="100%" stopColor="#ef4444" stopOpacity={0}/></linearGradient></defs>
                            <YAxis hide domain={[0, 280]}/>
                            <Area type="monotone" dataKey="ghost_speed" stroke="#ef4444" strokeWidth={2} fill="url(#grad)" isAnimationActive={false} />
                            <Area type="monotone" dataKey="real_speed" stroke="#555" strokeWidth={1} fill="transparent" strokeDasharray="3 3" isAnimationActive={false} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Controls */}
                <div className="h-8 flex items-center gap-6">
                    <button onClick={actions.togglePlay} className="w-8 h-8 rounded bg-white text-black flex items-center justify-center hover:scale-105 transition">
                        {isPlaying ? <Pause size={12} fill="black"/> : <Play size={12} fill="black" className="ml-0.5"/>}
                    </button>
                    <div className="flex-1 grid grid-cols-3 gap-4">
                        <Bar label="THR" val={frame.ghost_throttle || 0} col="bg-green-500"/>
                        <Bar label="BRK" val={frame.ghost_brake || 0} col="bg-red-500"/>
                        <Bar label="STR" val={Math.abs(frame.ghost_steer || 0)/4} col="bg-blue-500"/>
                    </div>
                </div>
            </div>

        </div>
      </div>
    </div>
  );
}

const Badge = ({ icon, label }) => (
    <div className="glass px-2 py-1 rounded flex items-center gap-2 text-[10px] font-medium text-white/80">
        <span className="opacity-70">{icon}</span>{label}
    </div>
);

const Bar = ({ label, val, col }) => (
    <div className="flex items-center gap-2 text-[9px] font-bold text-white/40">
        <span className="w-6">{label}</span>
        <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <motion.div className={`h-full ${col}`} animate={{ width: `${Math.min(val, 100)}%` }} transition={{ duration: 0.05 }} />
        </div>
    </div>
);

export default App;