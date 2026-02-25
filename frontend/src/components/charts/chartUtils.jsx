import { C, FONT, FONT_MONO } from '../../constants/theme';

export const SVG_H  = 200;
export const PAD    = { top:24, right:20, bottom:28, left:42 };
export const PLOT_H = SVG_H - PAD.top - PAD.bottom;

export function YGrid({ plotW }) {
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

export function ThresholdLine({ threshold, plotW }) {
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

export function RoundedBar({ x, y, w, h, r = 4, fill, opacity = 1 }) {
  if (h <= 0) return null;
  const safeR = Math.min(r, w / 2, h / 2);
  const d = `M${x+safeR},${y} h${w-2*safeR} a${safeR},${safeR} 0 0 1 ${safeR},${safeR}
             v${h-safeR} h${-w} v${-(h-safeR)} a${safeR},${safeR} 0 0 1 ${safeR},${-safeR}z`;
  return <path d={d} fill={fill} fillOpacity={opacity} />;
}

export function SvgTooltip({ tooltip }) {
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