import { useState, useRef, useCallback } from 'react';
import { C, FONT, FONT_MONO } from '../../constants/theme';
import { SVG_H, PAD, SvgTooltip } from './chartUtils';

export default function PGenuineGauge({ pGenuine, threshold, title }) {
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
          <rect x={PAD.left} y={trackY} width={plotW} height={trackH}
            rx={6} fill={C.surfaceAlt} stroke={C.border} strokeWidth={1} />
          {barW > 0 && (
            <rect x={PAD.left} y={trackY} width={barW} height={trackH}
              rx={6} fill={isAbove ? '#15803D':'#B91C1C'} fillOpacity={0.82}
              onMouseMove={handleMouse} style={{ cursor:'default' }} />
          )}
          <line x1={thX} y1={trackY - 10} x2={thX} y2={trackY + trackH + 10}
            stroke={C.blue} strokeWidth={2} strokeDasharray="5 3" />
          <rect x={thX - 26} y={trackY - 22} width={52} height={14} rx={3} fill={C.blueLight} />
          <text x={thX} y={trackY - 11} textAnchor="middle"
            fontSize={9} fontFamily={FONT_MONO} fill={C.blue} fontWeight="600">
            θ={threshold.toFixed(3)}
          </text>
          <text
            x={Math.min(PAD.left + barW + 5, PAD.left + plotW - 5)}
            y={trackY + trackH / 2 + 4}
            textAnchor="start" fontSize={10} fontFamily={FONT_MONO}
            fill={C.navy} fontWeight="700">
            {(pGenuine*100).toFixed(2)}%
          </text>
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