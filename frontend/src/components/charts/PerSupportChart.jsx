import { useState, useRef, useCallback } from 'react';
import { C, FONT } from '../../constants/theme';
import { SVG_H, PAD, PLOT_H, YGrid, ThresholdLine, RoundedBar, SvgTooltip } from './chartUtils';

export default function PerSupportChart({ perSupport, threshold, title }) {
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
                <text x={gx + groupW/2} y={SVG_H - 6} textAnchor="middle"
                  fontSize={10} fontFamily={FONT} fill={C.textSub}>
                  {s.support}
                </text>
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