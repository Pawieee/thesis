import { useState, useRef } from 'react';
import { C, FONT, FONT_MONO } from '../../constants/theme';
import { Ic } from '../icons/Icons';

export function Card({ children, style = {}, borderColor }) {
  return (
    <div style={{ background:C.surface, borderRadius:12, border:`1.5px solid ${borderColor||C.border}`, boxShadow:'0 1px 4px rgba(0,0,0,0.06)', overflow:'hidden', ...style }}>
      {children}
    </div>
  );
}

export function SectionHeader({ icon, title, subtitle, right }) {
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

export function Badge({ children, color = 'blue' }) {
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

export function StatBox({ label, value, valueColor, mono }) {
  return (
    <div style={{ background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:9, padding:'10px 12px', textAlign:'center' }}>
      <div style={{ fontFamily:FONT, fontSize:11, color:C.textSub, marginBottom:4, fontWeight:500 }}>{label}</div>
      <div style={{ fontFamily:mono?FONT_MONO:FONT, fontSize:mono?15:18, fontWeight:700, color:valueColor||C.text, letterSpacing:mono?0.3:-0.3 }}>{value}</div>
    </div>
  );
}

export function Divider({ label }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:10, margin:'18px 0 12px' }}>
      <div style={{ flex:1, height:1, background:C.borderLight }} />
      <span style={{ fontFamily:FONT, fontSize:11, color:C.textFaint, fontWeight:500, whiteSpace:'nowrap' }}>{label}</span>
      <div style={{ flex:1, height:1, background:C.borderLight }} />
    </div>
  );
}

export function UploadCard({ label, sublabel, file, preview, onChange, onClear, accent = 'amber' }) {
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