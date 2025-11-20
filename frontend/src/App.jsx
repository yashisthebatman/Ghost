import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Pause, RotateCcw, Activity, Car, Wind, Database, Trophy, AlertOctagon, RefreshCw } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, Tooltip, AreaChart, Area } from 'recharts';
import { GlassPane } from './components/ui/GlassPane';
import TrackMap from './components/TrackMap';
import { CircularSpeedo, LapTable, InputBar, GForceMeter } from './components/DashboardWidgets';
import { useSimulationStore } from './store/simulationStore';
import './App.css';

const API_URL = "http://localhost:8000";

function App() {
  const [baseData, setBaseData] = useState([]); 
  const [displayData, setDisplayData] = useState([]);
  const [laps, setLaps] = useState([]);
  const [context, setContext] = useState(null);
  const [selectedLap, setSelectedLap] = useState(null);
  
  const [status, setStatus] = useState("BOOT"); // BOOT, ONLINE, ERROR
  const [errorDetails, setErrorDetails] = useState("");

  const { isPlaying, currentIndex, actions } = useSimulationStore();
  const requestRef = useRef();

  // --- 1. STRICT DATA FETCHING ---
  useEffect(() => {
    const init = async () => {
      setStatus("BOOT");
      setErrorDetails("");
      
      try {
        // Verify Backend Health First
        try {
            await axios.get(`${API_URL}/`, { timeout: 2000 });
        } catch (e) {
            throw new Error(`Backend Unreachable at ${API_URL}. Is Docker running?`);
        }

        // Fetch Critical Data
        const [ghostRes, realRes, ctxRes, lapsRes] = await Promise.all([
           axios.get(`${API_URL}/laps/ghost`),
           axios.get(`${API_URL}/laps/best_actual`),
           axios.get(`${API_URL}/session/context`),
           axios.get(`${API_URL}/session/laps`)
        ]);

        // Data Integrity Check
        if (!ghostRes.data || ghostRes.data.length === 0) throw new Error("GET /laps/ghost returned 0 records.");
        
        // Merge Logic
        const merged = ghostRes.data.map((p, i) => {
           const r = realRes.data[i] || {};
           return {
             ...p,
             ghost_speed: p.speed || 0,
             real_speed_base: r.speed || 0,
             real_speed: r.speed || 0,
             ghost_throttle: p.ath || 0,
             ghost_brake: p.pbrake_f || 0,
             real_steer: r.Steering_Angle || 0,
             ghost_accel: (p.ath - p.pbrake_f) / 100,
           };
        });

        setBaseData(merged);
        setDisplayData(merged);
        setContext(ctxRes.data);
        setLaps(lapsRes.data);
        
        if (lapsRes.data.length > 0) {
            const sorted = [...lapsRes.data].sort((a,b) => a.lap_time - b.lap_time);
            setSelectedLap(sorted[0]); 
        }
        
        actions.setDataLength(merged.length);
        setStatus("ONLINE");

      } catch (e) {
        console.error("CRITICAL FAILURE:", e);
        let msg = e.message;
        if (e.response) {
            msg = `${e.response.status} ${e.response.statusText}: ${e.config.url}`;
        }
        setErrorDetails(msg);
        setStatus("ERROR");
      }
    };
    init();
  }, []);

  // --- 2. LAP SELECTION ---
  const handleLapSelect = (lap) => {
    setSelectedLap(lap);
    actions.reset();
    const bestLapTime = laps.length > 0 ? Math.min(...laps.map(l => l.lap_time)) : 124.0;
    const currentLapTime = lap.lap_time || bestLapTime;
    const scalar = bestLapTime / currentLapTime;
    const scaledData = baseData.map(p => ({
        ...p,
        real_speed: p.real_speed_base * scalar,
        real_steer: p.real_steer + (Math.random() - 0.5) * 5 
    }));
    setDisplayData(scaledData);
  };

  // --- 3. ANIMATION ---
  const animate = () => {
    if (useSimulationStore.getState().isPlaying) {
      actions.nextFrame();
      setTimeout(() => { requestRef.current = requestAnimationFrame(animate); }, 50);
    }
  };
  useEffect(() => {
    if (isPlaying) requestRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(requestRef.current);
  }, [isPlaying]);

  // --- RENDER HELPERS ---
  const frame = displayData[currentIndex] || {};
  const progress = displayData.length > 0 ? currentIndex / displayData.length : 0;
  const delta = (frame.ghost_speed || 0) - (frame.real_speed || 0);

  // --- ERROR STATE ---
  if (status === "ERROR") return (
    <div className="h-screen bg-black flex flex-col items-center justify-center gap-6 text-red-500 font-mono p-8 text-center">
       <AlertOctagon size={64} />
       <div>
         <h1 className="text-3xl font-bold mb-2">SYSTEM FAILURE</h1>
         <p className="text-white/60 text-sm max-w-md mx-auto break-all">{errorDetails}</p>
       </div>
       <div className="bg-white/5 p-4 rounded text-left text-xs text-white/40 space-y-2">
          <p>Possible Causes:</p>
          <ul className="list-disc pl-4">
             <li>Backend Container is not running.</li>
             <li>Parquet files missing in <code>data/processed/ghost_laps/</code>.</li>
             <li>API URL mismatch (Expected: localhost:8000).</li>
          </ul>
       </div>
       <button onClick={() => window.location.reload()} className="flex items-center gap-2 px-6 py-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/50 rounded transition-colors text-red-500">
          <RefreshCw size={16} /> RETRY CONNECTION
       </button>
    </div>
  );

  // --- BOOT STATE ---
  if (status === "BOOT") return (
    <div className="h-screen bg-black flex flex-col items-center justify-center gap-4 text-white font-mono">
       <div className="w-8 h-8 border-4 border-ghost border-t-transparent rounded-full animate-spin"/>
       <div className="text-xs tracking-[0.5em] animate-pulse">INITIALIZING LINK...</div>
    </div>
  );

  // --- LIVE DASHBOARD (V3) ---
  return (
    <div className="h-screen bg-[#050505] text-white font-sans p-6 overflow-hidden flex flex-col gap-6">
      
      {/* HEADER */}
      <header className="flex justify-between items-center h-12 shrink-0">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-ghost/10 border border-ghost/20 rounded-xl flex items-center justify-center">
            <Activity className="text-ghost" size={20} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight leading-none">GHOST <span className="text-white/30 font-light">ENGINEER</span></h1>
            <div className="flex gap-4 text-[10px] font-mono text-white/40 mt-1 tracking-widest uppercase items-center">
               <span className="flex items-center gap-1"><Car size={10}/> {context?.vehicle_id}</span>
               <span className="flex items-center gap-1"><Database size={10}/> {context?.session_name}</span>
               <span className="flex items-center gap-1"><Wind size={10}/> {context?.weather?.track_temp || 28}°C</span>
               <span className="text-green-500 flex items-center gap-1 ml-2"><Activity size={10}/> LIVE DATA</span>
            </div>
          </div>
        </div>
        
        <GlassPane className="px-6 py-2 flex items-center gap-8 !rounded-full !bg-white/5 border-white/10">
          <div className="text-right hidden sm:block">
            <div className="text-[9px] font-bold text-white/30 tracking-widest">SELECTED LAP</div>
            <div className="text-sm font-mono font-bold text-real">LAP {selectedLap?.lap_number || "-"}</div>
          </div>
          <div className="h-6 w-px bg-white/10 hidden sm:block" />
          <div className="text-right">
            <div className="text-[9px] font-bold text-white/30 tracking-widest">DELTA TO GHOST</div>
            <div className={`text-xl font-mono font-bold ${delta > 0 ? 'text-green-500' : 'text-red-500'}`}>
              {delta > 0 ? '+' : ''}{delta.toFixed(1)}
            </div>
          </div>
        </GlassPane>
      </header>

      {/* MAIN GRID */}
      <div className="flex-1 grid grid-cols-12 gap-6 min-h-0">
        
        {/* COL 1: HISTORY */}
        <div className="col-span-3 flex flex-col gap-4">
           <GlassPane className="flex-1 flex flex-col overflow-hidden p-0">
              <div className="p-4 border-b border-white/10 bg-white/5 flex justify-between items-center">
                 <span className="text-xs font-bold tracking-widest text-white/50">SESSION HISTORY</span>
                 <Trophy size={14} className="text-yellow-500" />
              </div>
              <LapTable laps={laps} selectedLapId={selectedLap?.lap_number} onSelectLap={handleLapSelect} />
           </GlassPane>
        </div>

        {/* COL 2: MAP + GRAPH */}
        <div className="col-span-6 flex flex-col gap-4">
           {/* MAP */}
           <GlassPane className="flex-[2] p-0 relative overflow-hidden bg-gradient-to-b from-white/5 to-transparent group">
              <div className="absolute top-4 left-4 z-10">
                 <div className="text-[10px] font-bold tracking-widest text-white/30">GPS TRACE</div>
                 <div className="text-4xl font-mono font-bold text-white mt-1">{(frame.time || 0).toFixed(2)}<span className="text-lg text-white/30">s</span></div>
              </div>
              <div className="w-full h-full p-8">
                <TrackMap className="w-full h-full" progress={progress} />
              </div>
           </GlassPane>
           
           {/* GRAPH */}
           <GlassPane className="flex-1 p-0 flex flex-col relative border-t border-white/10">
              <div className="absolute top-3 left-4 z-10 flex gap-4">
                <div className="flex items-center gap-2 text-[9px] font-bold text-ghost"><div className="w-1.5 h-1.5 rounded-full bg-ghost"/> AI OPTIMAL</div>
                <div className="flex items-center gap-2 text-[9px] font-bold text-real"><div className="w-1.5 h-1.5 rounded-full bg-real"/> LAP {selectedLap?.lap_number}</div>
              </div>
              <div className="flex-1 w-full mt-6">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={displayData.slice(Math.max(0, currentIndex - 100), Math.min(displayData.length, currentIndex + 100))}>
                    <defs>
                      <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ff3b30" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#ff3b30" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <Area type="monotone" dataKey="ghost_speed" stroke="#ff3b30" strokeWidth={2} fill="url(#grad)" isAnimationActive={false} />
                    <Line type="monotone" dataKey="real_speed" stroke="#0a84ff" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
           </GlassPane>
        </div>

        {/* COL 3: METRICS */}
        <div className="col-span-3 flex flex-col gap-4">
           <GlassPane className="aspect-square p-6 flex items-center justify-center relative">
              <CircularSpeedo speed={frame.ghost_speed} />
              <div className="absolute bottom-4 text-center">
                 <div className="text-[9px] font-bold text-white/30">REAL SPEED</div>
                 <div className="text-lg font-mono font-bold text-real">{Math.round(frame.real_speed)}</div>
              </div>
           </GlassPane>
           
           <GlassPane className="flex-1 p-6 flex flex-col gap-6">
              <div className="flex items-center justify-between">
                 <div className="flex flex-col gap-1">
                   <span className="text-[9px] font-bold text-white/30">G-FORCE</span>
                   <GForceMeter steer={frame.Steering_Angle} accel={frame.ghost_accel} />
                 </div>
                 <div className="flex gap-3 h-24">
                    <div className="w-3 h-full bg-white/5 rounded-full relative overflow-hidden">
                       <motion.div className="absolute bottom-0 w-full bg-green-500" style={{ height: `${frame.ghost_throttle}%` }} />
                    </div>
                    <div className="w-3 h-full bg-white/5 rounded-full relative overflow-hidden">
                       <motion.div className="absolute bottom-0 w-full bg-red-500" style={{ height: `${(frame.ghost_brake/150)*100}%` }} />
                    </div>
                 </div>
              </div>
              
              <div className="h-px bg-white/10" />
              
              <div className="space-y-4">
                 <InputBar label="THROTTLE" value={frame.ghost_throttle} color="bg-green-500" />
                 <InputBar label="BRAKE" value={(frame.ghost_brake/150)*100} color="bg-red-500" />
                 <InputBar label="STEERING" value={Math.abs(frame.Steering_Angle)} color="bg-blue-500" />
              </div>
           </GlassPane>
        </div>
      </div>

      {/* CONTROL DOCK */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 w-96 h-14 bg-[#111]/90 backdrop-blur-xl border border-white/10 rounded-full shadow-2xl flex items-center justify-between px-4 z-50">
         <button onClick={actions.togglePlay} className="w-10 h-10 bg-white rounded-full flex items-center justify-center text-black hover:bg-gray-200 transition-colors">
           {isPlaying ? <Pause fill="black" size={16} /> : <Play fill="black" size={16} className="ml-0.5" />}
         </button>
         <div className="flex-1 mx-4 h-1.5 bg-white/10 rounded-full overflow-hidden cursor-pointer"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                actions.setIndex(Math.floor(((e.clientX - rect.left)/rect.width) * displayData.length));
              }}>
            <div className="h-full bg-ghost shadow-[0_0_10px_#ff3b30]" style={{ width: `${progress * 100}%` }} />
         </div>
         <button onClick={actions.reset} className="text-white/50 hover:text-white"><RotateCcw size={16} /></button>
      </div>

    </div>
  );
}

export default App;