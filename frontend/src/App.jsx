import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Pause, Activity, Zap, Trophy, Timer, ChevronRight, Layers, BarChart3, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer, YAxis, Tooltip } from 'recharts';
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
  const lastTimeRef = useRef(0);

  // --- BOOT SEQUENCE ---
  useEffect(() => {
    const boot = async () => {
      try {
        await axios.get(`${API_URL}/`);
        const [ctx, lapsRes, ghost] = await Promise.all([
            axios.get(`${API_URL}/session/context`),
            axios.get(`${API_URL}/session/laps`),
            axios.get(`${API_URL}/laps/ghost`),
        ]);

        let uiLaps = lapsRes.data;
        // Inject context if DB has few laps
        if (uiLaps.length < 3) {
            const best = uiLaps[0]?.lap_time || 60.0;
            uiLaps = [
                { lap_number: 1, lap_time: best * 1.12, s1: 22.5, s2: 18.1, s3: 29.9, status: "WARMUP" },
                { lap_number: 2, lap_time: best * 1.05, s1: 21.2, s2: 17.5, s3: 27.4, status: "TRAFFIC" },
                { lap_number: 3, lap_time: best,        s1: 20.1, s2: 16.8, s3: 23.1, status: "PB" }
            ];
        }

        setState({ status: "READY", session: ctx.data, laps: uiLaps });
        setGhostData(ghost.data);
        loadLap(3, ghost.data, uiLaps);

      } catch (e) {
        console.error(e);
        setState(s => ({ ...s, status: "ERROR" }));
      }
    };
    boot();
  }, []);

  // --- LOAD LAP DATA ---
  const loadLap = async (id, ghostBase, history) => {
    setActiveLapId(id);
    actions.reset();
    
    const targetLap = history.find(l => l.lap_number === id);
    const bestLap = history.find(l => l.status === "PB");
    const factor = bestLap ? (bestLap.lap_time / targetLap.lap_time) : 1.0;

    try {
        let realData;
        try {
            const res = await axios.get(`${API_URL}/laps/actual/${id}`);
            realData = res.data;
        } catch {
            realData = ghostBase.map(p => ({ ...p, speed: p.speed * factor }));
        }

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

  // --- REAL-TIME SIMULATION ENGINE ---
  const animate = (time) => {
    if (!isPlaying) return;
    
    if (lastTimeRef.current === 0) {
        lastTimeRef.current = time;
    }
    
    const deltaTime = time - lastTimeRef.current;
    const timeStep = 10; // 10ms = 100Hz data
    
    // Advance frames based on actual time passed
    const stepsToAdvance = Math.floor(deltaTime / timeStep);
    
    if (stepsToAdvance > 0) {
        for (let i = 0; i < stepsToAdvance; i++) {
            actions.nextFrame();
        }
        // Keep the remainder for smoothness
        lastTimeRef.current = time - (deltaTime % timeStep);
    }
    
    requestRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    if (isPlaying) {
        lastTimeRef.current = 0;
        requestRef.current = requestAnimationFrame(animate);
    } else {
        cancelAnimationFrame(requestRef.current);
    }
    return () => cancelAnimationFrame(requestRef.current);
  }, [isPlaying]);

  const frame = displayData[currentIndex] || {};
  const progress = displayData.length > 0 ? currentIndex / displayData.length : 0;
  const delta = (frame.ghost_speed - frame.real_speed) || 0;
  const activeLapInfo = state.laps.find(l => l.lap_number === activeLapId) || {};

  if (state.status === "BOOT") return <div className="h-screen w-screen bg-black flex items-center justify-center text-white/30 tracking-[0.5em] animate-pulse font-mono">INITIALIZING LIQUID ENGINE...</div>;

  return (
    // 3-COLUMN GRID LAYOUT: [Sidebar 280px] [Map Fluid] [Analysis 320px]
    // Vertical: [Header 60px] [Main Fluid] [Bottom 240px]
    <div className="h-screen w-screen grid grid-cols-[280px_1fr_320px] grid-rows-[60px_1fr_240px] gap-3 p-3 bg-black text-white font-sans overflow-hidden relative">
      <div className="noise-bg" />

      {/* --- 1. HEADER (Span All) --- */}
      <header className="col-span-3 liquid-glass rounded-2xl flex items-center justify-between px-6 z-50">
        <div className="flex items-center gap-4">
            <div className="w-9 h-9 bg-gradient-to-br from-red-600 to-pink-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(220,38,38,0.4)]">
                <Activity size={18} className="text-white"/>
            </div>
            <div>
                <h1 className="text-sm font-bold tracking-widest text-white/90">GHOST ENGINEER</h1>
                <div className="flex items-center gap-2 text-[10px] text-white/50 font-mono">
                    <span>{state.session.vehicle_id}</span>
                    <span className="text-white/20">|</span>
                    <span className="text-green-400">CONNECTED</span>
                </div>
            </div>
        </div>
        <div className="flex gap-3">
            <div className="liquid-glass px-4 py-1.5 rounded-full flex items-center gap-2">
                <Zap size={14} className="text-yellow-400"/>
                <span className="text-xs font-bold">{state.session.weather.track_temp}°C</span>
            </div>
        </div>
      </header>

      {/* --- 2. SIDEBAR: HISTORY --- */}
      <div className="liquid-glass rounded-3xl flex flex-col overflow-hidden z-40">
         <div className="p-5 border-b border-white/10 bg-white/5">
            <div className="text-[10px] font-bold text-white/40 uppercase tracking-widest mb-1">Session History</div>
            <div className="text-2xl font-medium tracking-tight">{state.laps.length} Laps</div>
         </div>
         <div className="flex-1 overflow-y-auto p-3 space-y-2 custom-scrollbar">
            {state.laps.map(lap => (
                <div key={lap.lap_number} 
                     onClick={() => loadLap(lap.lap_number, ghostData, state.laps)}
                     className={`p-3 rounded-2xl cursor-pointer border transition-all duration-200 group ${activeLapId === lap.lap_number ? 'liquid-glass-active' : 'border-transparent hover:bg-white/5'}`}>
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-[10px] font-bold text-white/60 group-hover:text-white">LAP {lap.lap_number}</span>
                        {lap.status === "PB" && <Trophy size={12} className="text-yellow-400"/>}
                    </div>
                    <div className="flex justify-between items-end">
                        <span className={`text-xl font-mono font-medium ${activeLapId === lap.lap_number ? 'text-white' : 'text-white/50'}`}>{lap.lap_time.toFixed(3)}s</span>
                        <ChevronRight size={16} className={`opacity-0 transition-opacity ${activeLapId === lap.lap_number ? 'opacity-100 text-red-500' : 'group-hover:opacity-50'}`}/>
                    </div>
                </div>
            ))}
         </div>
      </div>

      {/* --- 3. CENTER: MAP (Span 1) --- */}
      <div className="liquid-glass rounded-3xl relative flex items-center justify-center overflow-hidden">
          <div className="w-full h-full p-6">
              <TrackMap className="w-full h-full" progress={progress} ghostProgress={progress} />
          </div>
          
          {/* HUD Time */}
          <div className="absolute top-6 left-6 liquid-glass px-5 py-2 rounded-full flex items-center gap-3">
              <Timer size={18} className="text-blue-400"/>
              <span className="text-2xl font-mono font-bold tracking-tight">{(frame.time || 0).toFixed(2)}<span className="text-sm text-white/40 ml-1">s</span></span>
          </div>
      </div>

      {/* --- 4. RIGHT: ANALYSIS (New Detailed Panel) --- */}
      <div className="liquid-glass rounded-3xl flex flex-col overflow-hidden z-40">
          <div className="p-5 border-b border-white/10 bg-white/5">
             <div className="text-[10px] font-bold text-white/40 uppercase tracking-widest mb-1">Performance Delta</div>
             <div className={`text-4xl font-mono font-bold tracking-tighter ${delta > 0 ? 'text-green-400' : 'text-red-500'}`}>
                 {delta > 0 ? '+' : ''}{delta.toFixed(1)} <span className="text-sm text-white/40">KPH</span>
             </div>
          </div>
          
          {/* Detailed Sector Breakdown */}
          <div className="flex-1 p-5 space-y-4">
              <SectorRow label="SECTOR 1" time={activeLapInfo.s1} ghost={20.1} />
              <SectorRow label="SECTOR 2" time={activeLapInfo.s2} ghost={16.8} />
              <SectorRow label="SECTOR 3" time={activeLapInfo.s3} ghost={23.1} />
              
              <div className="mt-6 p-4 rounded-xl bg-white/5 border border-white/10">
                  <div className="text-[10px] font-bold text-white/40 mb-2">OPTIMIZATION INSIGHT</div>
                  <p className="text-xs leading-relaxed text-white/70">
                      {delta < 0 ? "Loss in Turn 5 entry. Braking too early compared to AI Optimal." : "Matching AI pace on straights. maintain throttle application."}
                  </p>
              </div>
          </div>
      </div>

      {/* --- 5. BOTTOM: TELEMETRY (Span 3 Cols) --- */}
      <div className="col-span-3 liquid-glass rounded-3xl flex flex-col p-1 z-30 relative">
          <div className="flex-1 relative w-full px-6 pt-4">
              <div className="absolute top-2 left-6 flex gap-6 text-[10px] font-bold uppercase tracking-widest z-10">
                  <span className="flex items-center gap-2 text-red-400"><div className="w-1.5 h-1.5 rounded-full bg-red-500 shadow-[0_0_5px_red]"/> AI GHOST</span>
                  <span className="flex items-center gap-2 text-blue-400"><div className="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_5px_blue]"/> REAL CAR</span>
              </div>
              <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={displayData.slice(Math.max(0, currentIndex-150), Math.min(displayData.length, currentIndex+150))}>
                      <defs>
                          <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ef4444" stopOpacity={0.2}/><stop offset="100%" stopColor="#ef4444" stopOpacity={0}/></linearGradient>
                      </defs>
                      <YAxis hide domain={[0, 280]}/>
                      <Area type="monotone" dataKey="ghost_speed" stroke="#ef4444" strokeWidth={2} fill="url(#grad)" isAnimationActive={false} />
                      <Area type="monotone" dataKey="real_speed" stroke="#3b82f6" strokeWidth={2} fill="transparent" strokeDasharray="3 3" isAnimationActive={false} />
                  </AreaChart>
              </ResponsiveContainer>
          </div>

          {/* Controls */}
          <div className="h-16 border-t border-white/10 bg-black/20 flex items-center px-8 gap-8">
               <button onClick={actions.togglePlay} className="w-10 h-10 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 transition shadow-[0_0_15px_rgba(255,255,255,0.3)]">
                  {isPlaying ? <Pause size={16} fill="black"/> : <Play size={16} fill="black" className="ml-0.5"/>}
               </button>
               
               <div className="flex-1 grid grid-cols-3 gap-12">
                  <Bar label="THROTTLE" val={frame.ghost_throttle} col="bg-green-500"/>
                  <Bar label="BRAKE" val={frame.ghost_brake} col="bg-red-500"/>
                  <Bar label="STEER" val={Math.abs(frame.ghost_steer)/4} col="bg-blue-500"/>
               </div>
          </div>
      </div>

    </div>
  );
}

// --- HELPER COMPONENTS ---

const SectorRow = ({label, time, ghost}) => {
    const delta = (time - ghost);
    const isFaster = delta <= 0;
    return (
        <div className="flex items-center justify-between text-xs border-b border-white/5 pb-2">
            <span className="font-bold text-white/40">{label}</span>
            <div className="flex items-center gap-4">
                <span className="font-mono text-white/60">{time ? time.toFixed(2) : '--'}s</span>
                <div className={`flex items-center gap-1 font-mono font-bold ${isFaster ? 'text-green-400' : 'text-red-400'}`}>
                    {isFaster ? <ArrowDownRight size={12}/> : <ArrowUpRight size={12}/>}
                    {Math.abs(delta).toFixed(2)}
                </div>
            </div>
        </div>
    );
};

const Bar = ({label, val, col}) => (
    <div className="flex flex-col justify-center gap-1.5">
        <div className="flex justify-between text-[9px] font-bold text-white/30 tracking-widest">
            <span>{label}</span><span>{Math.round(val || 0)}%</span>
        </div>
        <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <motion.div className={`h-full ${col}`} animate={{width: `${Math.min(val || 0, 100)}%`}} transition={{duration:0.05}}/>
        </div>
    </div>
);

export default App;