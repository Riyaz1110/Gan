import { useState, useEffect, useRef } from "react";

// ── Matrix rain canvas ───────────────────────────────────────────
function MatrixCanvas() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    let animId;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const cols = Math.floor(canvas.width / 16);
    const drops = Array(cols).fill(1);
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&GAN01".split("");

    const draw = () => {
      ctx.fillStyle = "rgba(13,17,23,0.15)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = "14px monospace";

      drops.forEach((y, i) => {
        const ch = chars[Math.floor(Math.random() * chars.length)];
        // Alternate green shades for depth
        const bright = Math.random() > 0.95;
        ctx.fillStyle = bright ? "#c9ffd8" : (Math.random() > 0.5 ? "#3fb950" : "#1a7f37");
        ctx.fillText(ch, i * 16, y * 16);

        if (y * 16 > canvas.height && Math.random() > 0.975) drops[i] = 0;
        drops[i]++;
      });
      animId = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas ref={ref} style={canvasStyle} />;
}

// ── Credentials ──────────────────────────────────────────────────
const VALID_USER = "Raghav";
const VALID_PASS = "Raghav@123";

// ── Component ────────────────────────────────────────────────────
export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [shake, setShake] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    if (!username || !password) {
      trigger("Please fill in all fields.");
      return;
    }
    if (username !== VALID_USER || password !== VALID_PASS) {
      trigger("Invalid credentials. Access denied.");
      return;
    }

    setLoading(true);
    // Simulate auth handshake delay
    setTimeout(() => {
      setLoading(false);
      onLogin();
    }, 1200);
  };

  const trigger = (msg) => {
    setError(msg);
    setShake(true);
    setTimeout(() => setShake(false), 600);
  };

  return (
    <div style={wrapStyle}>
      <MatrixCanvas />

      {/* Glow orbs */}
      <div style={orb("rgba(63,185,80,0.18)", "-10%", "-10%", "500px")} />
      <div style={orb("rgba(88,166,255,0.10)", "60%", "60%", "400px")} />

      <div style={{ ...cardStyle, animation: shake ? "shake 0.5s ease" : "fadeUp 0.6s ease" }}>
        {/* Logo / Brand */}
        <div style={logoWrap}>
          <div style={shieldIcon}>
            {/* Simple SVG shield */}
            <svg width="38" height="44" viewBox="0 0 38 44" fill="none">
              <path d="M19 2L3 9v12c0 10.5 6.9 20.3 16 23 9.1-2.7 16-12.5 16-23V9L19 2z"
                fill="url(#sg)" stroke="#3fb950" strokeWidth="1.5" />
              <text x="50%" y="58%" dominantBaseline="middle" textAnchor="middle"
                fontSize="14" fontWeight="bold" fill="#fff" fontFamily="monospace">GAN</text>
              <defs>
                <linearGradient id="sg" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#21262d" />
                  <stop offset="100%" stopColor="#0d1117" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div>
            <h1 style={titleStyle}>CYBER SHIELD</h1>
            <p style={subtitleStyle}>GAN-Based Threat Detection Platform</p>
          </div>
        </div>

        {/* Divider */}
        <div style={dividerStyle} />

        <p style={accessLabel}>SECURE ACCESS PORTAL</p>

        {/* Form */}
        <form onSubmit={handleSubmit} autoComplete="off" style={{ width: "100%" }}>
          {/* Username */}
          <div style={fieldWrap}>
            <span style={fieldIcon}>👤</span>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              style={inputStyle}
              spellCheck={false}
            />
          </div>

          {/* Password */}
          <div style={fieldWrap}>
            <span style={fieldIcon}>🔒</span>
            <input
              type={showPass ? "text" : "password"}
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={inputStyle}
              spellCheck={false}
            />
            <button
              type="button"
              onClick={() => setShowPass((s) => !s)}
              style={eyeBtn}
              tabIndex={-1}
            >
              {showPass ? "🙈" : "👁️"}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div style={errorStyle}>
              <span>⚠️</span> {error}
            </div>
          )}

          {/* Submit */}
          <button type="submit" disabled={loading} style={loginBtn(loading)}>
            {loading ? (
              <span style={{ display: "flex", alignItems: "center", gap: "8px", justifyContent: "center" }}>
                <span style={spinnerStyle} /> Authenticating…
              </span>
            ) : (
              "▶  ACCESS SYSTEM"
            )}
          </button>
        </form>

        {/* Footer hint */}
        <p style={hintStyle}>
          Authorised personnel only &nbsp;·&nbsp; All access is monitored
        </p>
      </div>

      {/* Shake + fadeUp keyframes injected inline */}
      <style>{`
        @keyframes shake {
          0%,100%{transform:translateX(0)}
          20%{transform:translateX(-8px)}
          40%{transform:translateX(8px)}
          60%{transform:translateX(-6px)}
          80%{transform:translateX(6px)}
        }
        @keyframes fadeUp {
          from{opacity:0;transform:translateY(30px)}
          to{opacity:1;transform:translateY(0)}
        }
        @keyframes spin {
          to{transform:rotate(360deg)}
        }
        input::placeholder { color: #4d5561; }
        input:focus { outline: none; border-color: #3fb950 !important; box-shadow: 0 0 0 3px rgba(63,185,80,0.15); }
        button:not(:disabled):hover { filter: brightness(1.15); transform: translateY(-1px); }
        button { transition: all 0.2s ease; }
      `}</style>
    </div>
  );
}

// ── Styles ───────────────────────────────────────────────────────
const canvasStyle = {
  position: "fixed", top: 0, left: 0, width: "100%", height: "100%",
  zIndex: 0, opacity: 0.55,
};

const orb = (bg, top, left, size) => ({
  position: "fixed", top, left,
  width: size, height: size,
  background: `radial-gradient(circle, ${bg} 0%, transparent 70%)`,
  borderRadius: "50%", zIndex: 0, pointerEvents: "none",
  filter: "blur(40px)",
});

const wrapStyle = {
  minHeight: "100vh",
  background: "#0d1117",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontFamily: "'Segoe UI', system-ui, sans-serif",
  position: "relative",
  overflow: "hidden",
};

const cardStyle = {
  position: "relative",
  zIndex: 10,
  background: "rgba(22,27,34,0.85)",
  backdropFilter: "blur(20px)",
  border: "1px solid rgba(63,185,80,0.25)",
  borderRadius: "20px",
  padding: "44px 40px 36px",
  width: "100%",
  maxWidth: "420px",
  boxShadow: "0 0 0 1px rgba(63,185,80,0.08), 0 8px 64px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05)",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: "0",
};

const logoWrap = {
  display: "flex", alignItems: "center", gap: "14px", marginBottom: "8px",
};

const shieldIcon = {
  filter: "drop-shadow(0 0 10px rgba(63,185,80,0.6))",
};

const titleStyle = {
  margin: 0,
  fontSize: "22px",
  fontWeight: "800",
  color: "#e6edf3",
  letterSpacing: "3px",
};

const subtitleStyle = {
  margin: "2px 0 0",
  fontSize: "11px",
  color: "#3fb950",
  letterSpacing: "1px",
  textTransform: "uppercase",
};

const dividerStyle = {
  width: "100%",
  height: "1px",
  background: "linear-gradient(90deg, transparent, rgba(63,185,80,0.4), transparent)",
  margin: "20px 0 16px",
};

const accessLabel = {
  margin: "0 0 22px",
  fontSize: "10px",
  letterSpacing: "3px",
  color: "#58a6ff",
  textTransform: "uppercase",
  fontWeight: "600",
};

const fieldWrap = {
  position: "relative",
  display: "flex",
  alignItems: "center",
  marginBottom: "14px",
  width: "100%",
};

const fieldIcon = {
  position: "absolute",
  left: "14px",
  fontSize: "16px",
  pointerEvents: "none",
  zIndex: 1,
};

const inputStyle = {
  width: "100%",
  background: "rgba(13,17,23,0.8)",
  border: "1px solid #30363d",
  borderRadius: "10px",
  padding: "13px 14px 13px 44px",
  color: "#e6edf3",
  fontSize: "15px",
  boxSizing: "border-box",
  transition: "border-color 0.2s, box-shadow 0.2s",
};

const eyeBtn = {
  position: "absolute",
  right: "12px",
  background: "transparent",
  border: "none",
  cursor: "pointer",
  fontSize: "16px",
  padding: "4px",
  color: "#8b949e",
};

const errorStyle = {
  display: "flex",
  alignItems: "center",
  gap: "8px",
  background: "rgba(248,81,73,0.12)",
  border: "1px solid rgba(248,81,73,0.3)",
  borderRadius: "8px",
  color: "#f85149",
  fontSize: "13px",
  padding: "10px 14px",
  marginBottom: "14px",
  width: "100%",
  boxSizing: "border-box",
};

const loginBtn = (disabled) => ({
  width: "100%",
  padding: "14px",
  background: disabled
    ? "rgba(63,185,80,0.3)"
    : "linear-gradient(135deg, #238636 0%, #3fb950 100%)",
  border: "1px solid rgba(63,185,80,0.4)",
  borderRadius: "10px",
  color: "#fff",
  fontSize: "14px",
  fontWeight: "700",
  letterSpacing: "2px",
  cursor: disabled ? "not-allowed" : "pointer",
  marginTop: "6px",
  boxShadow: disabled ? "none" : "0 0 20px rgba(63,185,80,0.3)",
});

const spinnerStyle = {
  display: "inline-block",
  width: "14px",
  height: "14px",
  border: "2px solid rgba(255,255,255,0.3)",
  borderTopColor: "#fff",
  borderRadius: "50%",
  animation: "spin 0.8s linear infinite",
};

const hintStyle = {
  marginTop: "22px",
  fontSize: "11px",
  color: "#4d5561",
  textAlign: "center",
  letterSpacing: "0.5px",
};
