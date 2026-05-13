import { useEffect, useState } from "react";
import Login from "./Login";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend
} from "recharts";

// Severity colour palette
const SEVERITY_COLORS = {
  LOW:      "#3fb950",  // Green
  MEDIUM:   "#e3b341",  // Yellow
  HIGH:     "#f97316",  // Orange
  CRITICAL: "#f85149",  // Red
};

const severityColor = (sev) => SEVERITY_COLORS[sev] ?? "#8b949e";

// Custom dot for the line chart — coloured by severity
const SeverityDot = (props) => {
  const { cx, cy, payload } = props;
  const color = severityColor(payload?.severity);
  return (
    <circle
      cx={cx}
      cy={cy}
      r={5}
      fill={color}
      stroke={color}
      strokeWidth={1}
      style={{ filter: `drop-shadow(0 0 4px ${color})` }}
    />
  );
};

function App() {
  // ── All hooks must be declared unconditionally (Rules of Hooks) ──
  const [isLoggedIn,  setIsLoggedIn]  = useState(false);
  const [threats,     setThreats]     = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isTrained,   setIsTrained]   = useState(false);
  const [isTraining,  setIsTraining]  = useState(false);
  const [trainingLogs,setTrainingLogs]= useState([]);

  // Check backend status (only when logged in)
  useEffect(() => {
    if (!isLoggedIn) return;
    const checkStatus = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/status");
        const data = await res.json();
        setIsTrained(data.is_trained);
        setIsTraining(data.is_training);
        if (data.is_training) setTrainingLogs(data.progress.logs);
      } catch (err) {
        console.error("Failed to fetch status");
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, [isLoggedIn]);

  const handleTrain = async () => {
    try {
      await fetch("http://localhost:8000/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs: 5 })
      });
      setIsTraining(true);
    } catch (err) {
      console.error("Failed to start training");
    }
  };

  // WebSocket stream (only when trained)
  useEffect(() => {
    if (!isLoggedIn || !isTrained) return;

    const socket = new WebSocket("ws://localhost:8000/api/stream");
    socket.onopen  = () => setIsConnected(true);
    socket.onclose = () => setIsConnected(false);
    socket.onerror = () => setIsConnected(false);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) { console.error("Backend error:", data.error); return; }
      const now = new Date();
      data.date             = now.toISOString().split("T")[0];
      data.time             = now.toTimeString().split(" ")[0];
      data.datetime         = `${data.date} ${data.time}`;
      data.attack_probability = data.attack_prob;
      data.status           = data.is_attack ? "ATTACK DETECTED" : "NORMAL";
      data.id               = data.sample_id;
      setThreats((prev) => [...prev.slice(-30), data]);
    };

    return () => socket.close();
  }, [isLoggedIn, isTrained]);

  // ── Login gate ────────────────────────────────────────────────
  if (!isLoggedIn) {
    return <Login onLogin={() => setIsLoggedIn(true)} />;
  }

  const latest = threats[threats.length - 1];

  return (
    <div style={pageStyle}>
      <div style={headerContainerStyle}>
        <div>
          <h1>Live GAN-Based Cyber Attack Detection Dashboard</h1>
          <p>Real-time backend threat stream with date-time based attack visualization</p>
        </div>
        <div style={statusBadgeStyle(isConnected, isTrained)}>
          <span style={dotStyle(isConnected, isTrained)}></span>
          {isConnected ? "Live Stream Connected" : (isTrained ? "Live Stream Disconnected" : "Waiting for Training")}
        </div>
      </div>

      {!isTrained && (
        <div style={boxStyle}>
          <h2>Model Training Required</h2>
          <p>The GAN model must be trained before it can stream live threat predictions.</p>
          <button 
            onClick={handleTrain} 
            disabled={isTraining}
            style={buttonStyle(isTraining)}
          >
            {isTraining ? "Training in Progress..." : "Start Training (5 Epochs)"}
          </button>
          
          {isTraining && trainingLogs.length > 0 && (
            <div style={logBoxStyle}>
              {trainingLogs.map((log, i) => <div key={i}>{log}</div>)}
            </div>
          )}
        </div>
      )}

      <div style={{ ...cardContainer, opacity: isTrained ? 1 : 0.5 }}>
        <div style={cardStyle}>
          <h3>Date</h3>
          <h2>{latest?.date || "--"}</h2>
        </div>

        <div style={cardStyle}>
          <h3>Time</h3>
          <h2>{latest?.time || "--"}</h2>
        </div>

        <div style={cardStyle}>
          <h3>Attack Type</h3>
          <h2>{latest?.attack_type || "Waiting"}</h2>
        </div>

        <div style={{ ...cardStyle, borderColor: latest?.severity ? severityColor(latest.severity) : "#30363d", boxShadow: latest?.severity ? `0 0 12px ${severityColor(latest.severity)}55` : "none" }}>
          <h3>Severity</h3>
          <h2 style={{ color: latest?.severity ? severityColor(latest.severity) : "#3fb950", textShadow: latest?.severity ? `0 0 8px ${severityColor(latest.severity)}` : "none" }}>
            {latest?.severity || "LOW"}
          </h2>
        </div>

        <div style={cardStyle}>
          <h3>Status</h3>
          <h2>{latest?.status || "NORMAL"}</h2>
        </div>
      </div>

      <div style={boxStyle}>
        <h2>Live Attack Probability Graph</h2>

        {/* Severity legend */}
        <div style={severityLegendStyle}>
          {Object.entries(SEVERITY_COLORS).map(([label, color]) => (
            <span key={label} style={legendItemStyle(color)}>
              <span style={legendDotStyle(color)} />
              {label}
            </span>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={threats}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="datetime"
              angle={-25}
              textAnchor="end"
              height={80}
              tick={{ fill: "#8b949e", fontSize: 11 }}
            />
            <YAxis domain={[0, 1]} tick={{ fill: "#8b949e" }} />
            {/* Severity threshold reference lines */}
            <ReferenceLine y={0.90} stroke={SEVERITY_COLORS.CRITICAL} strokeDasharray="4 3" label={{ value: "CRITICAL", fill: SEVERITY_COLORS.CRITICAL, fontSize: 11, position: "insideTopRight" }} />
            <ReferenceLine y={0.75} stroke={SEVERITY_COLORS.HIGH}     strokeDasharray="4 3" label={{ value: "HIGH",     fill: SEVERITY_COLORS.HIGH,     fontSize: 11, position: "insideTopRight" }} />
            <ReferenceLine y={0.50} stroke={SEVERITY_COLORS.MEDIUM}   strokeDasharray="4 3" label={{ value: "MEDIUM",   fill: SEVERITY_COLORS.MEDIUM,   fontSize: 11, position: "insideTopRight" }} />
            <Tooltip
              contentStyle={{ background: "#161b22", border: "1px solid #30363d", borderRadius: "8px" }}
              labelStyle={{ color: "#c9d1d9" }}
              itemStyle={{ color: "#58a6ff" }}
              formatter={(value, name, props) => [
                <span style={{ color: severityColor(props.payload?.severity), fontWeight: "bold" }}>
                  {Number(value).toFixed(4)}
                </span>,
                <span>Attack Probability — <b style={{ color: severityColor(props.payload?.severity) }}>{props.payload?.severity}</b></span>
              ]}
            />
            <Line
              type="monotone"
              dataKey="attack_probability"
              stroke="#58a6ff"
              strokeWidth={2}
              dot={<SeverityDot />}
              activeDot={{ r: 7 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={boxStyle}>
        <h2>Live Threat Stream Data</h2>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th>Date</th>
              <th>Time</th>
              <th>Attack Type</th>
              <th>Probability</th>
              <th>Severity</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {threats.slice().reverse().map((item) => (
              <tr key={item.id} style={tableRowStyle(item.is_attack)}>
                <td style={tdStyle}>{item.date}</td>
                <td style={tdStyle}>{item.time}</td>
                <td style={tdStyle}>{item.attack_type}</td>
                <td style={{ ...tdStyle, fontFamily: "monospace", color: severityColor(item.severity) }}>
                  {Number(item.attack_probability).toFixed(4)}
                </td>
                <td style={tdStyle}>
                  <span style={severityBadgeStyle(item.severity)}>
                    {item.severity}
                  </span>
                </td>
                <td style={{ ...tdStyle, fontWeight: "bold", color: item.is_attack ? "#f85149" : "#3fb950" }}>
                  {item.status}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const pageStyle = {
  background: "#0d1117",
  color: "white",
  minHeight: "100vh",
  padding: "25px",
  fontFamily: "Arial"
};

const headerContainerStyle = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: "20px"
};

const statusBadgeStyle = (isConnected, isTrained) => {
  const color = isConnected ? "#3fb950" : (isTrained ? "#f85149" : "#d29922");
  const bg = isConnected ? "rgba(46, 160, 67, 0.15)" : (isTrained ? "rgba(248, 81, 73, 0.15)" : "rgba(210, 153, 34, 0.15)");
  const border = isConnected ? "rgba(46, 160, 67, 0.4)" : (isTrained ? "rgba(248, 81, 73, 0.4)" : "rgba(210, 153, 34, 0.4)");
  
  return {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    background: bg,
    color: color,
    padding: "10px 16px",
    borderRadius: "20px",
    fontWeight: "bold",
    border: `1px solid ${border}`
  };
};

const dotStyle = (isConnected, isTrained) => {
  const color = isConnected ? "#3fb950" : (isTrained ? "#f85149" : "#d29922");
  return {
    width: "10px",
    height: "10px",
    borderRadius: "50%",
    background: color,
    boxShadow: `0 0 8px ${color}`
  };
};

const cardContainer = {
  display: "grid",
  gridTemplateColumns: "repeat(5, 1fr)",
  gap: "15px",
  marginBottom: "25px"
};

const cardStyle = {
  background: "#161b22",
  padding: "18px",
  borderRadius: "12px",
  textAlign: "center",
  border: "1px solid #30363d"
};

const boxStyle = {
  background: "#161b22",
  padding: "20px",
  borderRadius: "12px",
  marginBottom: "25px",
  border: "1px solid #30363d"
};

const tableStyle = {
  width: "100%",
  borderCollapse: "collapse",
  textAlign: "center"
};

const tdStyle = {
  padding: "10px 12px",
  borderBottom: "1px solid #21262d"
};

const tableRowStyle = (isAttack) => ({
  background: isAttack ? "rgba(248, 81, 73, 0.05)" : "transparent",
  transition: "background 0.2s"
});

const severityBadgeStyle = (sev) => {
  const color = severityColor(sev);
  return {
    display: "inline-block",
    padding: "3px 10px",
    borderRadius: "12px",
    background: `${color}22`,
    color: color,
    border: `1px solid ${color}66`,
    fontWeight: "bold",
    fontSize: "12px",
    letterSpacing: "0.5px",
    boxShadow: `0 0 6px ${color}44`
  };
};

const severityLegendStyle = {
  display: "flex",
  gap: "20px",
  marginBottom: "12px",
  flexWrap: "wrap"
};

const legendItemStyle = (color) => ({
  display: "flex",
  alignItems: "center",
  gap: "6px",
  color: color,
  fontSize: "13px",
  fontWeight: "600",
  letterSpacing: "0.5px"
});

const legendDotStyle = (color) => ({
  width: "10px",
  height: "10px",
  borderRadius: "50%",
  background: color,
  boxShadow: `0 0 6px ${color}`
});

const buttonStyle = (disabled) => ({
  background: disabled ? "#21262d" : "#238636",
  color: disabled ? "#8b949e" : "white",
  border: "1px solid rgba(240, 246, 252, 0.1)",
  padding: "10px 20px",
  borderRadius: "6px",
  cursor: disabled ? "not-allowed" : "pointer",
  fontWeight: "bold",
  marginTop: "10px"
});

const logBoxStyle = {
  background: "#0d1117",
  padding: "15px",
  borderRadius: "6px",
  marginTop: "15px",
  fontFamily: "monospace",
  fontSize: "14px",
  color: "#8b949e",
  border: "1px solid #30363d",
  maxHeight: "150px",
  overflowY: "auto"
};

export default App;
