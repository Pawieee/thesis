import { C, FONT, FONT_MONO } from '../../constants/theme';
import { Ic } from '../icons/Icons';

export default function VoteDots({ perSupport }) {
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