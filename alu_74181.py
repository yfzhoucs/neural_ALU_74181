import gates.gate
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


class Alu74181(nn.Module):
    def __init__(self, state_dict):
        super(Alu74181, self).__init__()
        and_gates = []
        nand_gates = []
        nor_gates = []
        not_gates = []
        or_gates = []
        xor_gates = []
        for i in range(64):
            and_gates.append(gates.gate.Gate('and', state_dict))
        for i in range(1):
            nand_gates.append(gates.gate.Gate('nand', state_dict))
        for i in range(6):
            nor_gates.append(gates.gate.Gate('nor', state_dict))
        for i in range(16):
            not_gates.append(gates.gate.Gate('not', state_dict))
        for i in range(16):
            or_gates.append(gates.gate.Gate('or', state_dict))
        for i in range(4):
            xor_gates.append(gates.gate.Gate('xor', state_dict))
        self.and_gates = nn.ModuleList(and_gates)
        self.nand_gates = nn.ModuleList(nand_gates)
        self.nor_gates = nn.ModuleList(nor_gates)
        self.not_gates = nn.ModuleList(not_gates)
        self.or_gates = nn.ModuleList(or_gates)
        self.xor_gates = nn.ModuleList(xor_gates)

    def forward(self, cn, m, na0, na1, na2, na3, nb0, nb1, nb2, nb3, s0, s1, s2, s3):
        after_not0 = self.not_gates[0](m)
        after_not1 = self.not_gates[1](nb0)
        after_not2 = self.not_gates[2](nb1)
        after_not3 = self.not_gates[3](nb2)
        after_not4 = self.not_gates[4](nb3)

        after_and0 = self.and_gates[0](nb0, s0)
        after_and1 = self.and_gates[1](s1, after_not1)
        after_and3 = self._gate3In1Out_(self.and_gates[2], self.and_gates[3], after_not1, s2, na0)
        after_and5 = self._gate3In1Out_(self.and_gates[4], self.and_gates[5], na0, nb0, s3)
        after_and6 = self.and_gates[6](nb1, s0)
        after_and7 = self.and_gates[7](s1, after_not2)
        after_and9 = self._gate3In1Out_(self.and_gates[8], self.and_gates[9], after_not2, s2, na1)
        after_and11 = self._gate3In1Out_(self.and_gates[10], self.and_gates[11], na1, nb1, s3)        
        after_and12 = self.and_gates[12](nb2, s0)
        after_and13 = self.and_gates[13](s1, after_not3)
        after_and15 = self._gate3In1Out_(self.and_gates[14], self.and_gates[15], after_not3, s2, na2)
        after_and17 = self._gate3In1Out_(self.and_gates[16], self.and_gates[17], na2, nb2, s3)
        after_and18 = self.and_gates[18](nb3, s0)
        after_and19 = self.and_gates[19](s1, after_not4)
        after_and21 = self._gate3In1Out_(self.and_gates[20], self.and_gates[21], after_not4, s2, na3)
        after_and23 = self._gate3In1Out_(self.and_gates[22], self.and_gates[23], na3, nb3, s3)

        after_not9 = self.not_gates[9](self._gate3In1Out_(self.or_gates[0], self.or_gates[1], na0, after_and0, after_and1))
        after_nor0 = self.nor_gates[0](after_and3, after_and5)
        after_not10 = self.not_gates[10](self._gate3In1Out_(self.or_gates[2], self.or_gates[3], na1, after_and6, after_and7))
        after_nor1 = self.nor_gates[1](after_and9, after_and11)
        after_not11 = self.not_gates[11](self._gate3In1Out_(self.or_gates[4], self.or_gates[5], na2, after_and12, after_and13))
        after_nor2 = self.nor_gates[2](after_and15, after_and17)
        after_not12 = self.not_gates[12](self._gate3In1Out_(self.or_gates[6], self.or_gates[7], na3, after_and18, after_and19))
        after_nor3 = self.nor_gates[3](after_and21, after_and23)

        after_nand0 = self.nand_gates[0](cn, after_not0)
        after_not5 = self.not_gates[5](after_not9)
        after_and24 = self.and_gates[24](after_not0, after_not9)
        after_and26 = self._gate3In1Out_(self.and_gates[25], self.and_gates[26], after_not0, after_nor0, cn)
        after_not6 = self.not_gates[6](after_not10)
        after_and27 = self.and_gates[27](after_not0, after_not10)
        after_and29 = self._gate3In1Out_(self.and_gates[28], self.and_gates[29], after_not0, after_not9, after_nor1)
        after_and32 = self._gate4In1Out_(
            self.and_gates[30], self.and_gates[31], self.and_gates[32],
            after_not0, cn, after_nor0, after_nor1)
        after_not7 = self.not_gates[7](after_not11)
        after_and33 = self.and_gates[33](after_not0, after_not11)
        after_and35 = self._gate3In1Out_(self.and_gates[34], self.and_gates[35], after_not0, after_not10, after_nor2)
        after_and38 = self._gate4In1Out_(
            self.and_gates[36], self.and_gates[37], self.and_gates[38],
            after_not0, after_not9, after_nor1, after_nor2)
        after_and42 = self._gate5In1Out_(
            self.and_gates[39], self.and_gates[40], self.and_gates[41], self.and_gates[42],
            after_not0, cn, after_nor0, after_nor1, after_nor2)
        after_not8 = self.not_gates[8](after_not12)
        after_not15 = self.not_gates[15](self._gate4In1Out_(
            self.and_gates[57], self.and_gates[58], self.and_gates[59],
            after_nor0, after_nor1, after_nor2, after_nor3))
        after_and63 = self._gate5In1Out_(
            self.and_gates[60], self.and_gates[61], self.and_gates[62], self.and_gates[63],
            cn, after_nor0, after_nor1, after_nor2, after_nor3)
        after_and45 = self._gate4In1Out_(
            self.and_gates[43], self.and_gates[44], self.and_gates[45],
            after_not9, after_nor1, after_nor2, after_nor3)
        after_and47 = self._gate3In1Out_(self.and_gates[46], self.and_gates[47], after_not10, after_nor2, after_nor3)
        after_and48 = self.and_gates[48](after_not11, after_nor3)

        after_and50 = self.and_gates[50](after_not5, after_nor0)
        after_nor4 = self.nor_gates[4](after_and24, after_and26)
        after_and51 = self.and_gates[51](after_not6, after_nor1)
        after_not13 = self.not_gates[13](self._gate3In1Out_(
            self.or_gates[8], self.or_gates[9], after_and27, after_and29, after_and32))
        after_and52 = self.and_gates[52](after_not7, after_nor2)
        after_not14 = self.not_gates[14](self._gate4In1Out_(
            self.or_gates[10], self.or_gates[11], self.or_gates[12],
            after_and33, after_and35, after_and38, after_and42))
        after_and53 = self.and_gates[53](after_not8, after_nor3)
        after_or15 = self._gate4In1Out_(
            self.or_gates[13], self.or_gates[14], self.or_gates[15], 
            after_and45, after_and47, after_and48, after_not12)
        
        after_xor0 = self.xor_gates[0](after_nand0, after_and50)
        after_xor1 = self.xor_gates[1](after_nor4, after_and51)
        after_xor2 = self.xor_gates[2](after_not13, after_and52)
        after_xor3 = self.xor_gates[3](after_not14, after_and53)
        after_nor5 = self.nor_gates[5](after_and63, after_or15)

        after_and56 = self._gate4In1Out_(
            self.and_gates[54], self.and_gates[55], self.and_gates[56],
            after_xor0, after_xor1, after_xor2, after_xor3)
        
        nf0 = after_xor0
        nf1 = after_xor1
        a_eq_b = after_and56
        nf2 = after_xor2
        nf3 = after_xor3
        np = after_not15
        cn4 = after_nor5
        ng = after_or15

        return {
            'nf0': nf0, 
            'nf1': nf1, 
            'nf2': nf2, 
            'nf3': nf3, 
            'a_eq_b': a_eq_b, 
            'np': np, 
            'cn4': cn4, 
            'ng': ng
        }

    def _gate3In1Out_(self, c1, c2, x1, x2, x3):
        return c2(c1(x1, x2), x3)
    
    def _gate4In1Out_(self, c1, c2, c3, x1, x2, x3, x4):
        x12 = c1(x1, x2)
        x34 = c2(x3, x4)
        return c3(x12, x34)
    
    def _gate5In1Out_(self, c1, c2, c3, c4, x1, x2, x3, x4, x5):
        x12 = c1(x1, x2)
        x34 = c2(x3, x4)
        x1234 = c3(x12, x34)
        return c4(x1234, x5)


if __name__ == "__main__":
    name2path = {
        'and': torch.load('./ckpts/DatasetAndGate.pth'),
        'nand': torch.load('./ckpts/DatasetNandGate.pth'),
        'nor': torch.load('./ckpts/DatasetNorGate.pth'),
        'not': torch.load('./ckpts/DatasetNotGate.pth'),
        'or': torch.load('./ckpts/DatasetOrGate.pth'),
        'xor': torch.load('./ckpts/DatasetXorGate.pth'),
    }
    alu = Alu74181(name2path)
    with torch.no_grad():
        inp = {
            'cn': 1, 
            'm': 1, 
            'na0': 0, 
            'na1': 1, 
            'na2': 1, 
            'na3': 0, 
            'nb0': 0, 
            'nb1': 1, 
            'nb2': 1, 
            'nb3': 0, 
            's0': 1, 
            's1': 1, 
            's2': 1, 
            's3': 1
        }
        for key in inp:
            if inp[key] == 1:
                inp[key] = torch.ones(1, 1)
            else:
                inp[key] = torch.zeros(1, 1)

        outp = alu(**inp)
        outp = torch.cat([
            outp['nf0'], 
            outp['nf1'], 
            outp['nf2'], 
            outp['nf3'], 
            outp['a_eq_b'], 
            outp['np'], 
            outp['cn4'], 
            outp['ng']
        ], axis=1).numpy()
        outp[outp > 0.9] = 1
        outp[outp < 0.1] = 0
        print(outp)