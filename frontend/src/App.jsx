import { useState, useRef, useCallback } from 'react';

// ─── Design Tokens ────────────────────────────────────────────────────────────
const C = {
  bg:          '#F0F2F5',
  surface:     '#FFFFFF',
  surfaceAlt:  '#F7F8FA',
  border:      '#DDE1E9',
  borderLight: '#EEF0F4',
  navy:        '#0F1E36',
  navyMid:     '#1E3458',
  blue:        '#1D4ED8',
  blueLight:   '#EFF4FF',
  amber:       '#B45309',
  amberLight:  '#FFFBEB',
  amberBorder: '#FCD34D',
  green:       '#15803D',
  greenLight:  '#F0FDF4',
  greenBorder: '#86EFAC',
  red:         '#B91C1C',
  redLight:    '#FEF2F2',
  redBorder:   '#FCA5A5',
  violet:      '#5B21B6',
  violetLight: '#F5F3FF',
  text:        '#0F172A',
  textMid:     '#374151',
  textSub:     '#6B7280',
  textFaint:   '#9CA3AF',
};

const FONT      = `'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
const FONT_MONO = `'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace`;

const DATASETS = [
  { value: 'bhsig_bengali', label: 'BHSig-Bengali' },
  { value: 'bhsig_hindi',   label: 'BHSig-Hindi'   },
  { value: 'cedar',         label: 'CEDAR'          },
  { value: 'combined',      label: 'Combined (All)' },
];
const SPLITS = [
  { value: '70_15_15', label: '70 / 15 / 15' },
  { value: '65_18_18', label: '65 / 18 / 18' },
  { value: '60_20_20', label: '60 / 20 / 20' },
];

// ─── SVG Icons ────────────────────────────────────────────────────────────────
const Ic = {
  shield:   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
  upload:   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>,
  check:    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>,
  alert:    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>,
  loader:   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{width:'100%',height:'100%',animation:'spin 0.9s linear infinite'}}><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>,
  cpu:      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>,
  info:     <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>,
  clock:    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>,
  x:        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{width:'100%',height:'100%'}}><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>,
  settings: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>,
  chart:    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>,
  layers:   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" style={{width:'100%',height:'100%'}}><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>,
};

// ─── Primitives ───────────────────────────────────────────────────────────────
function Card({ children, style = {}, borderColor }) {
  return (
    <div style={{ background:C.surface, borderRadius:12, border:`1.5px solid ${borderColor||C.border}`, boxShadow:'0 1px 4px rgba(0,0,0,0.06)', overflow:'hidden', ...style }}>
      {children}
    </div>
  );
}

function SectionHeader({ icon, title, subtitle, right }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:10, padding:'13px 18px', borderBottom:`1px solid ${C.border}`, background:C.surfaceAlt }}>
      <span style={{ width:17, height:17, color:C.blue, flexShrink:0 }}>{icon}</span>
      <div style={{ flex:1 }}>
        <div style={{ fontFamily:FONT, fontWeight:600, fontSize:13, color:C.navy }}>{title}</div>
        {subtitle && <div style={{ fontFamily:FONT, fontSize:11, color:C.textSub, marginTop:1 }}>{subtitle}</div>}
      </div>
      {right && <div style={{ flexShrink:0 }}>{right}</div>}
    </div>
  );
}

function Badge({ children, color = 'blue' }) {
  const map = {
    blue:   { bg:C.blueLight,   text:C.blue   },
    green:  { bg:C.greenLight,  text:C.green  },
    red:    { bg:C.redLight,    text:C.red    },
    amber:  { bg:C.amberLight,  text:C.amber  },
    violet: { bg:C.violetLight, text:C.violet },
    gray:   { bg:'#F3F4F6',     text:'#374151' },
  };
  const { bg, text } = map[color] || map.blue;
  return <span style={{ fontFamily:FONT, fontWeight:600, fontSize:11, background:bg, color:text, borderRadius:5, padding:'2px 8px' }}>{children}</span>;
}

function StatBox({ label, value, valueColor, mono }) {
  return (
    <div style={{ background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:9, padding:'10px 12px', textAlign:'center' }}>
      <div style={{ fontFamily:FONT, fontSize:11, color:C.textSub, marginBottom:4, fontWeight:500 }}>{label}</div>
      <div style={{ fontFamily:mono?FONT_MONO:FONT, fontSize:mono?15:18, fontWeight:700, color:valueColor||C.text, letterSpacing:mono?0.3:-0.3 }}>{value}</div>
    </div>
  );
}

function Divider({ label }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:10, margin:'18px 0 12px' }}>
      <div style={{ flex:1, height:1, background:C.borderLight }} />
      <span style={{ fontFamily:FONT, fontSize:11, color:C.textFaint, fontWeight:500, whiteSpace:'nowrap' }}>{label}</span>
      <div style={{ flex:1, height:1, background:C.borderLight }} />
    </div>
  );
}

// ─── Upload Card ──────────────────────────────────────────────────────────────
function UploadCard({ label, sublabel, file, preview, onChange, onClear, accent = 'amber' }) {
  const inputRef = useRef();
  const [hover, setHover] = useState(false);
  const colors = {
    amber: { border:C.amber, activeBg:C.amberLight, activeBorder:C.amberBorder },
    rose:  { border:C.red,   activeBg:C.redLight,   activeBorder:C.redBorder   },
  }[accent];

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <span style={{ fontFamily:FONT, fontWeight:600, fontSize:13, color:C.navy }}>{label}</span>
        <span style={{ fontFamily:FONT, fontSize:11, color:C.textFaint }}>{sublabel}</span>
      </div>
      <div
        onClick={() => !file && inputRef.current.click()}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          position:'relative', border:`1.5px dashed ${file?colors.activeBorder:(hover?colors.border:C.border)}`,
          borderRadius:10, background:file?colors.activeBg:(hover?'#FAFBFF':C.surfaceAlt),
          minHeight:136, display:'flex', flexDirection:'column', alignItems:'center',
          justifyContent:'center', cursor:file?'default':'pointer',
          transition:'all 0.18s', overflow:'hidden', padding:12,
        }}
      >
        <input ref={inputRef} type="file" accept="image/*" style={{ display:'none' }} onChange={onChange} />
        {preview ? (
          <>
            <img src={preview} alt={label} style={{ maxHeight:96, maxWidth:'100%', objectFit:'contain', borderRadius:6, border:`1px solid ${C.border}` }} />
            <button
              onClick={e=>{ e.stopPropagation(); onClear(); }}
              style={{ position:'absolute',top:7,right:7,width:22,height:22,background:C.surface,border:`1px solid ${C.border}`,borderRadius:'50%',cursor:'pointer',display:'flex',alignItems:'center',justifyContent:'center',padding:4,color:C.textSub }}
            >
              <span style={{ width:11, height:11, display:'block' }}>{Ic.x}</span>
            </button>
            <span style={{ marginTop:6, fontFamily:FONT, fontSize:11, color:C.textSub }}>
              {file?.name?.length > 22 ? file.name.slice(0,20)+'…' : file?.name}
            </span>
          </>
        ) : (
          <div style={{ textAlign:'center', pointerEvents:'none' }}>
            <div style={{ width:24, height:24, margin:'0 auto 8px', color:hover?colors.border:C.textFaint }}>{Ic.upload}</div>
            <div style={{ fontFamily:FONT, fontSize:12, color:hover?colors.border:C.textSub, fontWeight:500 }}>Click to upload</div>
            <div style={{ fontFamily:FONT, fontSize:11, color:C.textFaint, marginTop:2 }}>PNG · JPG · JPEG</div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Pure SVG Charts (zero external dependencies, no useContext conflicts) ────
//
// All charts are built directly on SVG primitives. This eliminates the
// "Cannot read properties of null (reading 'useContext')" error that occurs
// when Recharts resolves a duplicate React instance in the Vite bundle.

const SVG_H    = 200;   // total svg height
const PAD      = { top:24, right:20, bottom:28, left:42 }; // inner chart padding
const PLOT_H   = SVG_H - PAD.top - PAD.bottom;

// Shared: horizontal grid lines + y-axis tick labels (0%, 25%, 50%, 75%, 100%)
function YGrid({ plotW }) {
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];
  return (
    <>
      {ticks.map(t => {
        const y = PAD.top + PLOT_H * (1 - t);
        return (
          <g key={t}>
            <line x1={PAD.left} y1={y} x2={PAD.left + plotW} y2={y}
              stroke={C.borderLight} strokeWidth={1} strokeDasharray={t===0?'none':'3 3'} />
            <text x={PAD.left - 5} y={y + 4} textAnchor="end"
              fontSize={9} fontFamily={FONT_MONO} fill={C.textFaint}>
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
    </>
  );
}

// Shared: threshold reference line + label
function ThresholdLine({ threshold, plotW }) {
  const y = PAD.top + PLOT_H * (1 - threshold);
  return (
    <g>
      <line x1={PAD.left} y1={y} x2={PAD.left + plotW} y2={y}
        stroke={C.blue} strokeWidth={1.5} strokeDasharray="6 3" />
      <rect x={PAD.left + plotW - 58} y={y - 15} width={56} height={14}
        rx={3} fill={C.blueLight} />
      <text x={PAD.left + plotW - 30} y={y - 4} textAnchor="middle"
        fontSize={9} fontFamily={FONT_MONO} fill={C.blue} fontWeight="600">
        θ={threshold.toFixed(3)}
      </text>
    </g>
  );
}

// Rounded-top rect helper (SVG path)
function RoundedBar({ x, y, w, h, r = 4, fill, opacity = 1 }) {
  if (h <= 0) return null;
  const safeR = Math.min(r, w / 2, h / 2);
  const d = `M${x+safeR},${y} h${w-2*safeR} a${safeR},${safeR} 0 0 1 ${safeR},${safeR}
             v${h-safeR} h${-w} v${-(h-safeR)} a${safeR},${safeR} 0 0 1 ${safeR},${-safeR}z`;
  return <path d={d} fill={fill} fillOpacity={opacity} />;
}

// Tooltip that follows mouse — pure HTML absolutely positioned
function SvgTooltip({ tooltip }) {
  if (!tooltip) return null;
  return (
    <div style={{
      position:'absolute', left: tooltip.x + 12, top: tooltip.y - 10,
      background: C.navy, border:`1px solid ${C.navyMid}`,
      borderRadius:8, padding:'8px 12px', pointerEvents:'none',
      fontFamily: FONT, zIndex: 50, whiteSpace:'nowrap',
      boxShadow:'0 4px 12px rgba(0,0,0,0.25)',
    }}>
      <div style={{ fontSize:11, fontWeight:600, color:'#E2E8F0', marginBottom:4 }}>{tooltip.label}</div>
      {tooltip.rows.map((r,i) => (
        <div key={i} style={{ fontSize:11, color: r.color||'#CBD5E1', marginBottom:1 }}>
          {r.name}: <strong style={{ color:'#F1F5F9' }}>{r.value}</strong>
        </div>
      ))}
    </div>
  );
}

// Chart 1 — Vertical grouped bars: P(Genuine) vs P(Forged) with threshold
function ProbDistChart({ pGenuine, pForged, threshold, title }) {
  const [tooltip, setTooltip] = useState(null);
  const svgRef = useRef(null);
  const bars = [
    { label:'P(Genuine)', value:pGenuine, fill:'#15803D' },
    { label:'P(Forged)',  value:pForged,  fill:'#B91C1C' },
  ];
  const svgW = 300;
  const plotW = svgW - PAD.left - PAD.right;
  const barW  = 52;
  const gap   = (plotW - bars.length * barW) / (bars.length + 1);

  const handleMouse = useCallback((e, bar) => {
    const rect = svgRef.current.getBoundingClientRect();
    setTooltip({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      label: bar.label,
      rows: [
        { name:'Value', value:`${(bar.value*100).toFixed(2)}%`, color: bar.fill },
        { name:'Threshold θ', value: threshold.toFixed(4), color:'#93C5FD' },
      ],
    });
  }, [threshold]);

  return (
    <div>
      <div style={{ fontFamily:FONT, fontSize:12, fontWeight:600, color:C.textMid, marginBottom:6 }}>{title}</div>
      <div style={{ position:'relative', display:'inline-block', width:'100%' }}>
        <svg ref={svgRef} viewBox={`0 0 ${svgW} ${SVG_H}`} width="100%" style={{ display:'block' }}
          onMouseLeave={() => setTooltip(null)}>
          <YGrid plotW={plotW} />
          <ThresholdLine threshold={threshold} plotW={plotW} />
          {bars.map((bar, i) => {
            const barH = PLOT_H * bar.value;
            const x = PAD.left + gap + i * (barW + gap);
            const y = PAD.top + PLOT_H - barH;
            return (
              <g key={bar.label} onMouseMove={e => handleMouse(e, bar)} style={{ cursor:'default' }}>
                <RoundedBar x={x} y={y} w={barW} h={barH} fill={bar.fill} opacity={0.85} />
                <text x={x + barW/2} y={y - 5} textAnchor="middle"
                  fontSize={9} fontFamily={FONT_MONO} fill={bar.fill} fontWeight="600">
                  {(bar.value*100).toFixed(1)}%
                </text>
                <text x={x + barW/2} y={SVG_H - 6} textAnchor="middle"
                  fontSize={10} fontFamily={FONT} fill={C.textSub}>
                  {bar.label}
                </text>
              </g>
            );
          })}
          {/* X axis baseline */}
          <line x1={PAD.left} y1={PAD.top + PLOT_H} x2={PAD.left + plotW} y2={PAD.top + PLOT_H}
            stroke={C.border} strokeWidth={1} />
        </svg>
        <SvgTooltip tooltip={tooltip} />
      </div>
    </div>
  );
}

// Chart 2 — Horizontal bar gauge: P(Genuine) vs threshold
function PGenuineGauge({ pGenuine, threshold, title }) {
  const [tooltip, setTooltip] = useState(null);
  const svgRef = useRef(null);
  const svgW = 300;
  const isAbove = pGenuine >= threshold;
  const trackH = 28;
  const trackY = SVG_H / 2 - trackH / 2;
  const plotW  = svgW - PAD.left - PAD.right;
  const barW   = plotW * pGenuine;
  const thX    = PAD.left + plotW * threshold;
  const xticks = [0, 0.25, 0.5, 0.75, 1.0];

  const handleMouse = useCallback((e) => {
    const rect = svgRef.current.getBoundingClientRect();
    setTooltip({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      label: 'P(Genuine)',
      rows: [
        { name:'Value',       value:`${(pGenuine*100).toFixed(2)}%`, color: isAbove ? '#86EFAC':'#FCA5A5' },
        { name:'Threshold θ', value: threshold.toFixed(4),           color:'#93C5FD' },
        { name:'Status',      value: isAbove ? 'Above θ ✓':'Below θ ✗', color: isAbove?'#86EFAC':'#FCA5A5' },
      ],
    });
  }, [pGenuine, threshold, isAbove]);

  return (
    <div>
      <div style={{ fontFamily:FONT, fontSize:12, fontWeight:600, color:C.textMid, marginBottom:6 }}>{title}</div>
      <div style={{ position:'relative', display:'inline-block', width:'100%' }}>
        <svg ref={svgRef} viewBox={`0 0 ${svgW} ${SVG_H}`} width="100%" style={{ display:'block' }}
          onMouseLeave={() => setTooltip(null)}>
          {/* x-axis ticks */}
          {xticks.map(t => {
            const x = PAD.left + plotW * t;
            return (
              <g key={t}>
                <line x1={x} y1={trackY - 4} x2={x} y2={trackY + trackH + 4}
                  stroke={C.borderLight} strokeWidth={1} strokeDasharray="3 3" />
                <text x={x} y={trackY + trackH + 16} textAnchor="middle"
                  fontSize={9} fontFamily={FONT_MONO} fill={C.textFaint}>
                  {(t*100).toFixed(0)}%
                </text>
              </g>
            );
          })}
          {/* Track background */}
          <rect x={PAD.left} y={trackY} width={plotW} height={trackH}
            rx={6} fill={C.surfaceAlt} stroke={C.border} strokeWidth={1} />
          {/* Filled bar */}
          {barW > 0 && (
            <rect x={PAD.left} y={trackY} width={barW} height={trackH}
              rx={6} fill={isAbove ? '#15803D':'#B91C1C'} fillOpacity={0.82}
              onMouseMove={handleMouse} style={{ cursor:'default' }} />
          )}
          {/* Threshold line */}
          <line x1={thX} y1={trackY - 10} x2={thX} y2={trackY + trackH + 10}
            stroke={C.blue} strokeWidth={2} strokeDasharray="5 3" />
          <rect x={thX - 26} y={trackY - 22} width={52} height={14} rx={3} fill={C.blueLight} />
          <text x={thX} y={trackY - 11} textAnchor="middle"
            fontSize={9} fontFamily={FONT_MONO} fill={C.blue} fontWeight="600">
            θ={threshold.toFixed(3)}
          </text>
          {/* Value label inside/beside bar */}
          <text
            x={Math.min(PAD.left + barW + 5, PAD.left + plotW - 5)}
            y={trackY + trackH / 2 + 4}
            textAnchor="start" fontSize={10} fontFamily={FONT_MONO}
            fill={C.navy} fontWeight="700">
            {(pGenuine*100).toFixed(2)}%
          </text>
          {/* Y-label */}
          <text x={PAD.left - 6} y={trackY + trackH/2 + 4} textAnchor="end"
            fontSize={10} fontFamily={FONT} fill={C.textSub}>
            P(G)
          </text>
        </svg>
        <SvgTooltip tooltip={tooltip} />
      </div>
    </div>
  );
}

// Chart 3 — Grouped vertical bars per support
function PerSupportChart({ perSupport, threshold, title }) {
  const [tooltip, setTooltip] = useState(null);
  const svgRef = useRef(null);
  const svgW   = 380;
  const plotW  = svgW - PAD.left - PAD.right;
  const groups = perSupport.length;
  const barW   = 18;
  const barGap = 4;
  const groupW = barW * 2 + barGap;
  const groupGap = (plotW - groups * groupW) / (groups + 1);

  const seriesDef = [
    { key:'p_genuine', label:'P(Genuine)', fill:'#15803D' },
    { key:'p_forged',  label:'P(Forged)',  fill:'#B91C1C' },
  ];

  const handleMouse = useCallback((e, support, seriesLabel, value, fill) => {
    const rect = svgRef.current.getBoundingClientRect();
    setTooltip({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      label: support,
      rows: [
        { name: seriesLabel, value:`${(value*100).toFixed(2)}%`, color: fill },
        { name:'Threshold θ', value: threshold.toFixed(4), color:'#93C5FD' },
      ],
    });
  }, [threshold]);

  return (
    <div>
      <div style={{ fontFamily:FONT, fontSize:12, fontWeight:600, color:C.textMid, marginBottom:6 }}>{title}</div>
      {/* Legend */}
      <div style={{ display:'flex', gap:14, marginBottom:6 }}>
        {seriesDef.map(s => (
          <div key={s.key} style={{ display:'flex', alignItems:'center', gap:5 }}>
            <div style={{ width:10, height:10, borderRadius:2, background:s.fill, opacity:0.85 }} />
            <span style={{ fontFamily:FONT, fontSize:11, color:C.textSub }}>{s.label}</span>
          </div>
        ))}
      </div>
      <div style={{ position:'relative', display:'inline-block', width:'100%' }}>
        <svg ref={svgRef} viewBox={`0 0 ${svgW} ${SVG_H}`} width="100%" style={{ display:'block' }}
          onMouseLeave={() => setTooltip(null)}>
          <YGrid plotW={plotW} />
          <ThresholdLine threshold={threshold} plotW={plotW} />
          {perSupport.map((s, gi) => {
            const gx = PAD.left + groupGap + gi * (groupW + groupGap);
            return (
              <g key={s.support}>
                {seriesDef.map((sd, si) => {
                  const val  = s[sd.key];
                  const barH = PLOT_H * val;
                  const bx   = gx + si * (barW + barGap);
                  const by   = PAD.top + PLOT_H - barH;
                  return (
                    <g key={sd.key}
                      onMouseMove={e => handleMouse(e, s.support, sd.label, val, sd.fill)}
                      style={{ cursor:'default' }}>
                      <RoundedBar x={bx} y={by} w={barW} h={barH} fill={sd.fill} opacity={0.82} />
                    </g>
                  );
                })}
                {/* Group label */}
                <text x={gx + groupW/2} y={SVG_H - 6} textAnchor="middle"
                  fontSize={10} fontFamily={FONT} fill={C.textSub}>
                  {s.support}
                </text>
                {/* Prediction badge */}
                <text x={gx + groupW/2} y={PAD.top - 8} textAnchor="middle"
                  fontSize={9} fontFamily={FONT} fontWeight="600"
                  fill={s.prediction==='GENUINE' ? C.green : C.red}>
                  {s.prediction==='GENUINE' ? '✓' : '✗'}
                </text>
              </g>
            );
          })}
          <line x1={PAD.left} y1={PAD.top + PLOT_H} x2={PAD.left + plotW} y2={PAD.top + PLOT_H}
            stroke={C.border} strokeWidth={1} />
        </svg>
        <SvgTooltip tooltip={tooltip} />
      </div>
    </div>
  );
}

// ─── Vote Indicators ──────────────────────────────────────────────────────────
function VoteDots({ perSupport }) {
  return (
    <div style={{ display:'flex', gap:10 }}>
      {perSupport.map((s, i) => {
        const isG = s.prediction === 'GENUINE';
        return (
          <div key={i} style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:5, flex:1 }}>
            <div style={{
              width:44, height:44, borderRadius:'50%',
              background:isG?C.greenLight:C.redLight,
              border:`2px solid ${isG?C.greenBorder:C.redBorder}`,
              display:'flex', alignItems:'center', justifyContent:'center',
              color:isG?C.green:C.red,
            }}>
              <span style={{ width:18, height:18, display:'block' }}>{isG?Ic.check:Ic.alert}</span>
            </div>
            <span style={{ fontFamily:FONT, fontWeight:600, fontSize:12, color:isG?C.green:C.red }}>{s.support}</span>
            <span style={{ fontFamily:FONT_MONO, fontSize:11, color:C.textMid }}>P(G)={(s.p_genuine * 100).toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Results Panels ───────────────────────────────────────────────────────────
function ProposedPanel({ data }) {
  const isG = data.prediction === 'GENUINE';
  return (
    <Card borderColor={isG?C.greenBorder:C.redBorder}>
      {/* Verdict header */}
      <div style={{
        padding:'16px 20px',
        background:isG?C.greenLight:C.redLight,
        borderBottom:`1px solid ${isG?C.greenBorder:C.redBorder}`,
        display:'flex', alignItems:'center', justifyContent:'space-between',
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:12 }}>
          <span style={{ width:24, height:24, color:isG?C.green:C.red }}>{isG?Ic.check:Ic.alert}</span>
          <div>
            <div style={{ fontFamily:FONT, fontSize:11, fontWeight:600, color:isG?C.green:C.red, letterSpacing:0.3, marginBottom:2 }}>
              PROPOSED MODEL · 3× K=1 Majority Vote
            </div>
            <div style={{ fontFamily:FONT, fontSize:26, fontWeight:800, color:isG?C.green:C.red, lineHeight:1, letterSpacing:-0.5 }}>
              {data.prediction}
            </div>
          </div>
        </div>
        <div style={{ textAlign:'right' }}>
          <div style={{ fontFamily:FONT, fontSize:11, color:C.textSub, marginBottom:2 }}>Vote Confidence</div>
          <div style={{ fontFamily:FONT_MONO, fontSize:24, fontWeight:700, color:C.navy }}>{data.vote_confidence}%</div>
        </div>
      </div>

      <div style={{ padding:'18px 20px' }}>
        {/* Vote dots */}
        <div style={{ fontFamily:FONT, fontSize:12, fontWeight:600, color:C.textSub, marginBottom:10 }}>Individual K=1 Comparisons</div>
        <VoteDots perSupport={data.per_support} />

        {/* Stats */}
        <Divider label="Probability Scores" />
        <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:9, marginBottom:4 }}>
          <StatBox label="Avg P(Genuine)"  value={(data.avg_p_genuine * 100).toFixed(2)} mono valueColor={C.green} />
          <StatBox label="Avg P(Forged)"   value={(data.avg_p_forged * 100).toFixed(2)}  mono valueColor={C.red}   />
          <StatBox label="Threshold (θ)"   value={(data.threshold * 100).toFixed(2)}     mono valueColor={C.blue}  />
          <StatBox label="Vote Tally"      value={`${data.per_support?.filter(s=>s.prediction==='GENUINE').length ?? '—'}/3`} mono />
        </div>

        <Divider label="Score Visualization" />
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:4 }}>
          <ProbDistChart pGenuine={data.avg_p_genuine} pForged={data.avg_p_forged} threshold={data.threshold} title="Probability Distribution" />
          <PGenuineGauge pGenuine={data.avg_p_genuine} threshold={data.threshold} title="P(Genuine) vs Threshold" />
        </div>

        <Divider label="Per-Support Comparison" />
        <div style={{ marginBottom:4 }}>
          <PerSupportChart perSupport={data.per_support} threshold={data.threshold} title="Per-Support Scores" />
        </div>

        {/* Table */}
        <Divider label="Detailed Results Table" />
        <div style={{ borderRadius:8, overflow:'hidden', border:`1px solid ${C.border}` }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr style={{ background:C.surfaceAlt }}>
                {['Reference','P(Genuine)','P(Forged)','Vote'].map(h => (
                  <th key={h} style={{ padding:'8px 14px', fontFamily:FONT, fontSize:11, fontWeight:600, color:C.textSub, textAlign:'left', borderBottom:`1px solid ${C.border}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.per_support.map((s,i) => (
                <tr key={i} style={{ borderBottom:`1px solid ${C.borderLight}`, background:i%2?C.surfaceAlt:C.surface }}>
                  <td style={{ padding:'9px 14px', fontFamily:FONT, fontWeight:600, fontSize:12, color:C.navy }}>{s.support}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.green }}>{(s.p_genuine * 100).toFixed(2)}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.red   }}>{(s.p_forged * 100).toFixed(2)}</td>
                  <td style={{ padding:'9px 14px' }}><Badge color={s.prediction==='GENUINE'?'green':'red'}>{s.prediction}</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Technical details */}
        <Divider label="Technical Pipeline" />
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10 }}>
          {[
            { title:'Feature Extraction', rows:[
              ['Support Features','1024-d × 3'],
              ['Query Features','1024-d × 1'],
              ['Combined Input','2048-d'],
              ['Preprocessing','Otsu → Invert → Resize → Normalize'],
            ]},
            { title:'Metric Computation', rows:[
              ['Operation','concat [support, query]'],
              ['MLP Output','Logit → Sigmoid'],
              ['Final Output','P(Genuine) ∈ [0, 1]'],
              ['Decision Rule',`P(G) ≥ ${(data.threshold * 100).toFixed(2)} → GENUINE`],
            ]},
          ].map(({ title, rows }) => (
            <div key={title} style={{ background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:8, padding:'12px 14px' }}>
              <div style={{ fontFamily:FONT, fontSize:11, fontWeight:700, color:C.blue, marginBottom:8 }}>{title}</div>
              {rows.map(([k,v]) => (
                <div key={k} style={{ display:'flex', justifyContent:'space-between', padding:'3px 0', borderBottom:`1px solid ${C.borderLight}`, gap:8 }}>
                  <span style={{ fontFamily:FONT, fontSize:11, color:C.textSub, flexShrink:0 }}>{k}</span>
                  <span style={{ fontFamily:FONT_MONO, fontSize:11, color:C.textMid, fontWeight:500, textAlign:'right' }}>{v}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

function BaselinePanel({ data, proposedData }) {
  const isG = data.prediction === 'GENUINE';
  return (
    <Card>
      {/* Verdict header */}
      <div style={{
        padding:'16px 20px',
        background:isG?C.greenLight:C.redLight,
        borderBottom:`1px solid ${isG?C.greenBorder:C.redBorder}`,
        display:'flex', alignItems:'center', justifyContent:'space-between',
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:12 }}>
          <span style={{ width:24, height:24, color:isG?C.green:C.red }}>{isG?Ic.check:Ic.alert}</span>
          <div>
            <div style={{ fontFamily:FONT, fontSize:11, fontWeight:600, color:isG?C.green:C.red, letterSpacing:0.3, marginBottom:2 }}>
              BASELINE MODEL · DenseNet Classifier
            </div>
            <div style={{ fontFamily:FONT, fontSize:26, fontWeight:800, color:isG?C.green:C.red, lineHeight:1, letterSpacing:-0.5 }}>
              {data.prediction}
            </div>
          </div>
        </div>
        <div style={{ textAlign:'right' }}>
          <div style={{ fontFamily:FONT, fontSize:11, color:isG?C.green:C.red, marginBottom:2 }}>Confidence</div>
          <div style={{ fontFamily:FONT_MONO, fontSize:24, fontWeight:700, color:isG?C.green:C.red }}>{data.confidence.toFixed(2)}%</div>
        </div>
      </div>

      <div style={{ padding:'18px 20px' }}>
        <div style={{ background:C.blueLight, border:`1px solid ${C.border}`, borderRadius:8, padding:'10px 14px', fontFamily:FONT, fontSize:12, color:C.textMid, marginBottom:16, lineHeight:1.55 }}>
          The baseline DenseNet classifier is evaluated on the <strong>query image only</strong> — it does not use the 3 support references.
        </div>

        {/* Stats */}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:9, marginBottom:4 }}>
          <StatBox label="Verdict"    value={data.prediction}                        mono={false} valueColor={isG?C.green:C.red} />
          <StatBox label="P(Genuine)" value={(data.p_genuine * 100).toFixed(2)}       mono valueColor={C.green} />
          <StatBox label="P(Forged)"  value={(data.p_forged * 100).toFixed(2)}        mono valueColor={C.red}   />
          <StatBox label="Confidence" value={(data.confidence).toFixed(2) + '%'}      mono />
        </div>

        {/* Charts */}
        <Divider label="Score Visualization" />
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:4 }}>
          <ProbDistChart pGenuine={data.p_genuine} pForged={data.p_forged} threshold={data.threshold} title="Baseline Probability Distribution" />
          <PGenuineGauge pGenuine={data.p_genuine} threshold={data.threshold} title="Baseline P(Genuine) vs Threshold" />
        </div>

        {/* Probability Table */}
        <Divider label="Probability Breakdown" />
        <div style={{ borderRadius:8, overflow:'hidden', border:`1px solid ${C.border}` }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr style={{ background:C.surfaceAlt }}>
                {['Metric','Value','vs Threshold'].map(h => (
                  <th key={h} style={{ padding:'8px 14px', fontFamily:FONT, fontSize:11, fontWeight:600, color:C.textSub, textAlign:'left', borderBottom:`1px solid ${C.border}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['P(Genuine)',   (data.p_genuine * 100).toFixed(2),   `${((data.p_genuine - data.threshold) * 100).toFixed(2)}`],
                ['P(Forged)',    (data.p_forged * 100).toFixed(2),    `${((data.p_forged - data.threshold) * 100).toFixed(2)}`],
                ['Threshold θ', (data.threshold * 100).toFixed(2),   '—'],
              ].map(([label,val,diff],i) => (
                <tr key={label} style={{ borderBottom:`1px solid ${C.borderLight}`, background:i%2?C.surfaceAlt:C.surface }}>
                  <td style={{ padding:'9px 14px', fontFamily:FONT, fontWeight:500, fontSize:12, color:C.navy }}>{label}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.textMid }}>{val}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:diff.startsWith('-')?C.red:(diff==='—'?C.textFaint:C.green) }}>{diff}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

      </div>
    </Card>
  );
}

// ─── Model Comparison Panel ───────────────────────────────────────────────────
function ModelComparisonPanel({ proposed, baseline }) {
  const speedup = (proposed.processing_time / baseline.processing_time).toFixed(1);
  const proposedIsG = proposed.prediction === 'GENUINE';
  const baselineIsG = baseline.prediction === 'GENUINE';
  const agree = proposed.prediction === baseline.prediction;

  return (
    <Card>
      <SectionHeader icon={Ic.chart} title="Model Comparison" subtitle="Proposed (tDCBAM) vs Baseline (DenseNet Classifier)" />
      <div style={{ padding:'18px 20px' }}>

        {/* Agreement banner */}
        <div style={{
          background: agree ? C.greenLight : C.amberLight,
          border: `1px solid ${agree ? C.greenBorder : C.amberBorder}`,
          borderRadius: 9, padding:'10px 16px',
          display:'flex', alignItems:'center', gap:10, marginBottom:18,
        }}>
          <span style={{ width:16, height:16, color: agree ? C.green : C.amber, flexShrink:0 }}>
            {agree ? Ic.check : Ic.alert}
          </span>
          <span style={{ fontFamily:FONT, fontSize:13, fontWeight:600, color: agree ? C.green : '#78350F' }}>
            {agree
              ? `Both models agree — verdict is ${proposed.prediction}`
              : `Models disagree — Proposed: ${proposed.prediction} · Baseline: ${baseline.prediction}`}
          </span>
        </div>

        {/* Side-by-side verdict + time */}
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginBottom:18 }}>
          {[
            { label:'Proposed · tDCBAM',    prediction: proposed.prediction, time: proposed.processing_time, color: proposedIsG ? C.green : C.red, detail:'3× K=1 Majority Vote · 4 images' },
            { label:'Baseline · DenseNet',  prediction: baseline.prediction, time: baseline.processing_time, color: baselineIsG ? C.green : C.red, detail:'Single query · no support set' },
          ].map(({ label, prediction, time, color, detail }) => (
            <div key={label} style={{ background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:10, padding:'14px 16px' }}>
              <div style={{ fontFamily:FONT, fontSize:11, fontWeight:600, color:C.textSub, marginBottom:6 }}>{label}</div>
              <div style={{ fontFamily:FONT, fontSize:20, fontWeight:800, color, marginBottom:6 }}>{prediction}</div>
              <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                <span style={{ width:12, height:12, color:C.textFaint, display:'block' }}>{Ic.clock}</span>
                <span style={{ fontFamily:FONT_MONO, fontSize:12, color:C.textMid }}>{time}s</span>
              </div>
              <div style={{ fontFamily:FONT, fontSize:11, color:C.textFaint, marginTop:4 }}>{detail}</div>
            </div>
          ))}
        </div>

        {/* Stats comparison table */}
        <Divider label="Score Comparison" />
        <div style={{ borderRadius:8, overflow:'hidden', border:`1px solid ${C.border}`, marginBottom:18 }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr style={{ background:C.surfaceAlt }}>
                {['Metric','Proposed','Baseline'].map(h => (
                  <th key={h} style={{ padding:'8px 14px', fontFamily:FONT, fontSize:11, fontWeight:600, color:C.textSub, textAlign:'left', borderBottom:`1px solid ${C.border}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['Verdict',       proposed.prediction,                              baseline.prediction],
                ['P(Genuine)',    (proposed.avg_p_genuine * 100).toFixed(2),        (baseline.p_genuine * 100).toFixed(2)],
                ['P(Forged)',     (proposed.avg_p_forged * 100).toFixed(2),         (baseline.p_forged * 100).toFixed(2)],
                ['Confidence',   `${proposed.vote_confidence}%`,                   `${baseline.confidence.toFixed(2)}%`],
                ['Threshold θ',  (proposed.threshold * 100).toFixed(2),            (baseline.threshold * 100).toFixed(2)],
                ['Process Time', `${proposed.processing_time}s`,                   `${baseline.processing_time}s`],
              ].map(([label, pVal, bVal], i) => (
                <tr key={label} style={{ borderBottom:`1px solid ${C.borderLight}`, background:i%2?C.surfaceAlt:C.surface }}>
                  <td style={{ padding:'9px 14px', fontFamily:FONT, fontWeight:500, fontSize:12, color:C.navy }}>{label}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.textMid }}>{pVal}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.textMid }}>{bVal}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Speed trade-off note */}
        <Divider label="Speed vs Accuracy Trade-off" />
        <div style={{ background:C.amberLight, border:`1px solid ${C.amberBorder}`, borderRadius:8, padding:'12px 16px', fontFamily:FONT, fontSize:12, color:'#78350F', lineHeight:1.65 }}>
          ⚡ The Baseline is <strong>{speedup}× faster</strong> ({baseline.processing_time}s vs {proposed.processing_time}s).
          The Proposed model processes <strong>4 images</strong> (3 supports + 1 query) using metric learning for more robust verification,
          at the cost of higher inference time. The Baseline classifies the query image alone with no reference set.
        </div>

      </div>
    </Card>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [dataset, setDataset] = useState('bhsig_bengali');
  const [split, setSplit]     = useState('70_15_15');
  const [supports, setSupports]         = useState([null,null,null]);
  const [previews, setPreviews]         = useState([null,null,null]);
  const [queryFile, setQueryFile]       = useState(null);
  const [queryPreview, setQueryPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);
  const resultsRef = useRef(null);

  const handleSupport = (i) => (e) => {
    const f = e.target.files[0]; if (!f) return;
    const ns=[...supports], np=[...previews];
    ns[i]=f; np[i]=URL.createObjectURL(f);
    setSupports(ns); setPreviews(np);
  };
  const clearSupport = (i) => () => {
    const ns=[...supports], np=[...previews];
    ns[i]=null; np[i]=null;
    setSupports(ns); setPreviews(np);
  };
  const handleQuery = (e) => {
    const f=e.target.files[0]; if (!f) return;
    setQueryFile(f); setQueryPreview(URL.createObjectURL(f));
  };

  const allReady = !supports.includes(null) && !!queryFile;

  const verify = async () => {
    setLoading(true); setResult(null); setError(null);
    const fd = new FormData();
    fd.append('dataset', dataset); fd.append('split', split);
    fd.append('support_file_1', supports[0]);
    fd.append('support_file_2', supports[1]);
    fd.append('support_file_3', supports[2]);
    fd.append('query_file', queryFile);
    try {
      const r = await fetch('http://localhost:8000/verify', { method:'POST', body:fd });
      if (!r.ok) { const d = await r.json(); throw new Error(d.detail||'Server error'); }
      setResult(await r.json());
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior:'smooth', block:'start' }), 120);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const proposed = result?.proposed;
  const baseline = result?.baseline;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
        html, body, #root { width:100%; min-height:100vh; }
        body { background:${C.bg}; font-family:${FONT}; color:${C.text}; }
        @keyframes spin { to { transform:rotate(360deg); } }
        @keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
        .fade-up { animation:fadeUp 0.45s cubic-bezier(0.16,1,0.3,1) both; }
        select { appearance:none; -webkit-appearance:none; }
        button, select, input { font-family:${FONT}; }
        ::-webkit-scrollbar { width:6px; height:6px; }
        ::-webkit-scrollbar-track { background:${C.bg}; }
        ::-webkit-scrollbar-thumb { background:${C.border}; border-radius:3px; }
      `}</style>

      <div style={{ width:'100%', minHeight:'100vh', background:C.bg }}>

        {/* ── Sticky Nav ──────────────────────────────────────────────── */}
        <header style={{
          width:'100%', background:C.navy, borderBottom:`1px solid ${C.navyMid}`,
          padding:'0 32px', display:'flex', alignItems:'center', height:58, gap:14,
          position:'sticky', top:0, zIndex:100, boxShadow:'0 2px 12px rgba(0,0,0,0.20)',
        }}>
          <span style={{ width:22, height:22, color:'#60A5FA', flexShrink:0 }}>{Ic.shield}</span>
          <span style={{ fontFamily:FONT, fontWeight:700, fontSize:17, color:'#F1F5F9', letterSpacing:-0.3 }}>
            Signature Verification
          </span>
          <span style={{ fontFamily:FONT_MONO, fontSize:11, color:'#60A5FA' }}>tDCBAM · 3×K=1</span>
          <div style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:7 }}>
            <div style={{ width:7, height:7, borderRadius:'50%', background:'#4ADE80', boxShadow:'0 0 6px #4ADE80' }} />
            <span style={{ fontFamily:FONT, fontSize:12, color:'#94A3B8' }}>API Online</span>
          </div>
        </header>

        {/* ── Main Content — full width ─────────────────────────────── */}
        <div style={{ width:'100%', padding:'28px 32px 56px' }}>

          {/* Page heading */}
          <div style={{ marginBottom:22 }}>
            <h1 style={{ fontFamily:FONT, fontWeight:700, fontSize:22, color:C.navy, letterSpacing:-0.4, marginBottom:6 }}>
              Test Set Protocol — K=1 Episodic Evaluation
            </h1>
            <p style={{ fontFamily:FONT, fontSize:13, color:C.textSub, lineHeight:1.65, maxWidth:820 }}>
              Upload <strong>3 genuine support signatures</strong> and <strong>1 query signature</strong>.
              The model performs 3 independent K=1 comparisons using a learned metric and returns a <strong>majority vote</strong> as the final verdict.
              The learned decision threshold minimises the Equal Error Rate on the validation set.
            </p>
          </div>

          {/* ── Config + Upload row ─────────────────────────────────── */}
          <div style={{ display:'grid', gridTemplateColumns:'260px 1fr', gap:18, marginBottom:20, alignItems:'start' }}>

            {/* Left: Config sidebar */}
            <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
              <Card>
                <SectionHeader icon={Ic.settings} title="Configuration" />
                <div style={{ padding:'16px' }}>
                  <div style={{ marginBottom:14 }}>
                    <label style={{ display:'block', fontFamily:FONT, fontWeight:600, fontSize:12, color:C.textMid, marginBottom:6 }}>Dataset</label>
                    <div style={{ position:'relative' }}>
                      <select value={dataset} onChange={e=>setDataset(e.target.value)} style={{ width:'100%', padding:'9px 34px 9px 11px', background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:8, fontFamily:FONT, fontSize:13, color:C.text, cursor:'pointer', outline:'none' }}>
                        {DATASETS.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                      </select>
                      <span style={{ position:'absolute', right:10, top:'50%', transform:'translateY(-50%)', color:C.textFaint, pointerEvents:'none' }}>▾</span>
                    </div>
                  </div>
                  <div>
                    <label style={{ display:'block', fontFamily:FONT, fontWeight:600, fontSize:12, color:C.textMid, marginBottom:7 }}>Train / Val / Test Split</label>
                    <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
                      {SPLITS.map(s => (
                        <div key={s.value} onClick={() => setSplit(s.value)} style={{
                          padding:'8px 12px', borderRadius:8,
                          border:`1.5px solid ${split===s.value?C.blue:C.border}`,
                          background:split===s.value?C.blueLight:C.surfaceAlt,
                          cursor:'pointer', transition:'all 0.15s',
                          display:'flex', alignItems:'center', justifyContent:'space-between',
                        }}>
                          <span style={{ fontFamily:FONT_MONO, fontSize:13, fontWeight:600, color:split===s.value?C.blue:C.textMid }}>{s.label}</span>
                          {split===s.value && <Badge color="blue">Active</Badge>}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>

              <Card>
                <SectionHeader icon={Ic.layers} title="Architecture" />
                <div style={{ padding:'12px 16px' }}>
                  {[
                    ['Backbone',   'DenseNet-121 + CBAM'],
                    ['Feat. Dim',  '1024-d embeddings'],
                    ['Metric MLP', '2048 → 512 → 1'],
                    ['Activation', 'Sigmoid'],
                    ['Loss Fn',    'BCEWithLogitsLoss'],
                    ['Threshold',  'Learned (min. EER)'],
                    ['Final Vote', '≥ 2/3 → GENUINE'],
                  ].map(([k,v]) => (
                    <div key={k} style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline', padding:'5px 0', borderBottom:`1px solid ${C.borderLight}` }}>
                      <span style={{ fontFamily:FONT, fontSize:12, color:C.textSub }}>{k}</span>
                      <span style={{ fontFamily:FONT_MONO, fontSize:11, color:C.textMid, fontWeight:500, textAlign:'right', maxWidth:130 }}>{v}</span>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            {/* Right: Upload panel */}
            <Card>
              <SectionHeader icon={Ic.upload} title="Upload Signatures" subtitle="3 genuine supports + 1 query to verify" />
              <div style={{ padding:'20px' }}>

                {/* Supports */}
                <div style={{ marginBottom:18 }}>
                  <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:12 }}>
                    <Badge color="amber">Support Set · 3× K=1</Badge>
                    <span style={{ fontFamily:FONT, fontSize:12, color:C.textSub }}>Genuine reference signatures from the test user</span>
                  </div>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:14 }}>
                    {[0,1,2].map(i => (
                      <UploadCard key={i} label={`Support ${i+1}`} sublabel="Genuine Ref" file={supports[i]} preview={previews[i]} onChange={handleSupport(i)} onClear={clearSupport(i)} accent="amber" />
                    ))}
                  </div>
                </div>

                <Divider label="Query Signature" />

                {/* Query Signature + Status + CTA stacked */}
                <div style={{ display:'flex', flexDirection:'column', gap:14 }}>

                  {/* Query upload — full width */}
                  <div>
                    <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:12 }}>
                      <Badge color="red">Query</Badge>
                      <span style={{ fontFamily:FONT, fontSize:12, color:C.textSub }}>Genuine or forged</span>
                    </div>
                    <UploadCard label="Query Signature" sublabel="Test Image" file={queryFile} preview={queryPreview} onChange={handleQuery} onClear={()=>{setQueryFile(null);setQueryPreview(null);}} accent="rose" />
                  </div>

                  {/* Upload status grid */}
                  <div style={{ background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:10, padding:'12px 16px' }}>
                    <div style={{ fontFamily:FONT, fontSize:12, fontWeight:600, color:C.textMid, marginBottom:10 }}>Upload Status</div>
                    <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:8 }}>
                      {[
                        {label:'Support 1', ready:!!supports[0]},
                        {label:'Support 2', ready:!!supports[1]},
                        {label:'Support 3', ready:!!supports[2]},
                        {label:'Query',     ready:!!queryFile},
                      ].map(({label,ready}) => (
                        <div key={label} style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:5 }}>
                          <div style={{
                            width:34, height:34, borderRadius:'50%',
                            background:ready?C.greenLight:'#F3F4F6',
                            border:`2px solid ${ready?C.greenBorder:C.border}`,
                            display:'flex', alignItems:'center', justifyContent:'center',
                            color:ready?C.green:C.textFaint, transition:'all 0.2s',
                          }}>
                            {ready
                              ? <span style={{width:14,height:14,display:'block'}}>{Ic.check}</span>
                              : <span style={{fontFamily:FONT_MONO,fontSize:13,fontWeight:700}}>?</span>
                            }
                          </div>
                          <span style={{ fontFamily:FONT, fontSize:11, color:ready?C.green:C.textFaint, fontWeight:ready?600:400 }}>{label}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* CTA */}
                  <button
                    onClick={verify}
                    disabled={loading||!allReady}
                    onMouseEnter={e=>{ if(!loading&&allReady) e.currentTarget.style.background=C.blue; }}
                    onMouseLeave={e=>{ if(!loading&&allReady) e.currentTarget.style.background=C.navy; }}
                    style={{
                      width:'100%', padding:'13px 20px',
                      background:loading||!allReady?'#E5E7EB':C.navy,
                      color:loading||!allReady?C.textFaint:'#FFF',
                      border:'none', borderRadius:10,
                      fontFamily:FONT, fontWeight:700, fontSize:14,
                      cursor:loading||!allReady?'not-allowed':'pointer',
                      display:'flex', alignItems:'center', justifyContent:'center', gap:10,
                      transition:'background 0.18s ease',
                      boxShadow:loading||!allReady?'none':'0 4px 14px rgba(15,30,54,0.24)',
                    }}
                  >
                    {loading
                      ? <><span style={{width:17,height:17,display:'inline-block'}}>{Ic.loader}</span> Analyzing…</>
                      : <><span style={{width:17,height:17,display:'inline-block'}}>{Ic.shield}</span> Run Verification</>
                    }
                  </button>

                  {/* Error */}
                  {error && (
                    <div style={{ background:C.redLight, border:`1px solid ${C.redBorder}`, borderRadius:8, padding:'10px 14px', display:'flex', gap:8, alignItems:'flex-start' }}>
                      <span style={{ width:15,height:15,color:C.red,flexShrink:0,marginTop:1 }}>{Ic.alert}</span>
                      <div>
                        <div style={{ fontFamily:FONT, fontWeight:600, fontSize:12, color:C.red, marginBottom:2 }}>Verification Failed</div>
                        <div style={{ fontFamily:FONT, fontSize:12, color:'#7F1D1D' }}>{error}</div>
                      </div>
                    </div>
                  )}

                </div>
              </div>
            </Card>
          </div>

          {/* ── Results ─────────────────────────────────────────────── */}
          {result && (
            <div ref={resultsRef} className="fade-up">
              <div style={{ display:'flex', alignItems:'center', gap:12, marginBottom:16 }}>
                <span style={{ width:20,height:20,color:C.blue }}>{Ic.chart}</span>
                <h2 style={{ fontFamily:FONT, fontWeight:700, fontSize:19, color:C.navy, letterSpacing:-0.3 }}>Verification Results</h2>
                <div style={{ flex:1, height:1, background:C.border }} />
                <Badge color={proposed.prediction==='GENUINE'?'green':'red'}>Final: {proposed.prediction}</Badge>
              </div>

              {/* Side-by-side panels */}
              <div style={{
                display:'grid',
                gridTemplateColumns: baseline?.available ? '1fr 1fr' : '1fr',
                gap:18,
                alignItems:'start',
              }}>
                <ProposedPanel data={proposed} />
                {baseline?.available && <BaselinePanel data={baseline} proposedData={proposed} />}
              </div>

              {/* Model comparison container */}
              {baseline?.available && (
                <div style={{ marginTop:18 }}>
                  <ModelComparisonPanel proposed={proposed} baseline={baseline} />
                </div>
              )}

              {result.baseline && !baseline?.available && (
                <div style={{ marginTop:12, background:C.surface, border:`1px solid ${C.border}`, borderRadius:10, padding:'12px 16px', display:'flex', gap:8, alignItems:'center' }}>
                  <span style={{ width:15,height:15,color:C.amber }}>{Ic.info}</span>
                  <span style={{ fontFamily:FONT, fontSize:13, color:C.textSub }}>Baseline model is unavailable for the <strong>Combined</strong> dataset.</span>
                </div>
              )}
            </div>
          )}

          {/* Footer */}
          <div style={{ marginTop:40, borderTop:`1px solid ${C.border}`, paddingTop:16, display:'flex', justifyContent:'space-between', flexWrap:'wrap', gap:6 }}>
            <span style={{ fontFamily:FONT, fontSize:12, color:C.textFaint }}>tDCBAM · Signature Verification · Test Protocol Tool</span>
            <span style={{ fontFamily:FONT, fontSize:12, color:C.textFaint }}>Matches proposed notebook evaluation exactly · K=1 Episodic Protocol</span>
          </div>
        </div>
      </div>
    </>
  );
}