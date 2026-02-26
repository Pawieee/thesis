import { useState, useRef, useCallback } from 'react';
import { C, FONT, FONT_MONO } from '../../constants/theme';
import { SVG_H, PAD, PLOT_H, YGrid, ThresholdLine, RoundedBar, SvgTooltip } from './chartUtils';

export default function ProbDistChart({ pGenuine, pForged, threshold, title }) {
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
          <line x1={PAD.left} y1={PAD.top + PLOT_H} x2={PAD.left + plotW} y2={PAD.top + PLOT_H}
            stroke={C.border} strokeWidth={1} />
        </svg>
        <SvgTooltip tooltip={tooltip} />
      </div>
    </div>
  );
}