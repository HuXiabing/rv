# Generated from Program.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ProgramParser import ProgramParser
else:
    from ProgramParser import ProgramParser

# This class defines a complete listener for a parse tree produced by ProgramParser.
class ProgramListener(ParseTreeListener):

    # Enter a parse tree produced by ProgramParser#prog.
    def enterProg(self, ctx:ProgramParser.ProgContext):
        pass

    # Exit a parse tree produced by ProgramParser#prog.
    def exitProg(self, ctx:ProgramParser.ProgContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst.
    def enterInst(self, ctx:ProgramParser.InstContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst.
    def exitInst(self, ctx:ProgramParser.InstContext):
        pass


    # Enter a parse tree produced by ProgramParser#label.
    def enterLabel(self, ctx:ProgramParser.LabelContext):
        pass

    # Exit a parse tree produced by ProgramParser#label.
    def exitLabel(self, ctx:ProgramParser.LabelContext):
        pass


    # Enter a parse tree produced by ProgramParser#rfmt.
    def enterRfmt(self, ctx:ProgramParser.RfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#rfmt.
    def exitRfmt(self, ctx:ProgramParser.RfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#ifmt.
    def enterIfmt(self, ctx:ProgramParser.IfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#ifmt.
    def exitIfmt(self, ctx:ProgramParser.IfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#mfmt.
    def enterMfmt(self, ctx:ProgramParser.MfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#mfmt.
    def exitMfmt(self, ctx:ProgramParser.MfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#bfmt.
    def enterBfmt(self, ctx:ProgramParser.BfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#bfmt.
    def exitBfmt(self, ctx:ProgramParser.BfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#jfmt.
    def enterJfmt(self, ctx:ProgramParser.JfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#jfmt.
    def exitJfmt(self, ctx:ProgramParser.JfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#ufmt.
    def enterUfmt(self, ctx:ProgramParser.UfmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#ufmt.
    def exitUfmt(self, ctx:ProgramParser.UfmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#amofmt.
    def enterAmofmt(self, ctx:ProgramParser.AmofmtContext):
        pass

    # Exit a parse tree produced by ProgramParser#amofmt.
    def exitAmofmt(self, ctx:ProgramParser.AmofmtContext):
        pass


    # Enter a parse tree produced by ProgramParser#pseudo.
    def enterPseudo(self, ctx:ProgramParser.PseudoContext):
        pass

    # Exit a parse tree produced by ProgramParser#pseudo.
    def exitPseudo(self, ctx:ProgramParser.PseudoContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_bz.
    def enterInst_bz(self, ctx:ProgramParser.Inst_bzContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_bz.
    def exitInst_bz(self, ctx:ProgramParser.Inst_bzContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_bgtle.
    def enterInst_bgtle(self, ctx:ProgramParser.Inst_bgtleContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_bgtle.
    def exitInst_bgtle(self, ctx:ProgramParser.Inst_bgtleContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_j.
    def enterInst_j(self, ctx:ProgramParser.Inst_jContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_j.
    def exitInst_j(self, ctx:ProgramParser.Inst_jContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_jr.
    def enterInst_jr(self, ctx:ProgramParser.Inst_jrContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_jr.
    def exitInst_jr(self, ctx:ProgramParser.Inst_jrContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_mv.
    def enterInst_mv(self, ctx:ProgramParser.Inst_mvContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_mv.
    def exitInst_mv(self, ctx:ProgramParser.Inst_mvContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_neg.
    def enterInst_neg(self, ctx:ProgramParser.Inst_negContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_neg.
    def exitInst_neg(self, ctx:ProgramParser.Inst_negContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_nop.
    def enterInst_nop(self, ctx:ProgramParser.Inst_nopContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_nop.
    def exitInst_nop(self, ctx:ProgramParser.Inst_nopContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_ret.
    def enterInst_ret(self, ctx:ProgramParser.Inst_retContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_ret.
    def exitInst_ret(self, ctx:ProgramParser.Inst_retContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fence.
    def enterInst_fence(self, ctx:ProgramParser.Inst_fenceContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fence.
    def exitInst_fence(self, ctx:ProgramParser.Inst_fenceContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fencetso.
    def enterInst_fencetso(self, ctx:ProgramParser.Inst_fencetsoContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fencetso.
    def exitInst_fencetso(self, ctx:ProgramParser.Inst_fencetsoContext):
        pass


    # Enter a parse tree produced by ProgramParser#fence_single.
    def enterFence_single(self, ctx:ProgramParser.Fence_singleContext):
        pass

    # Exit a parse tree produced by ProgramParser#fence_single.
    def exitFence_single(self, ctx:ProgramParser.Fence_singleContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fencei.
    def enterInst_fencei(self, ctx:ProgramParser.Inst_fenceiContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fencei.
    def exitInst_fencei(self, ctx:ProgramParser.Inst_fenceiContext):
        pass


    # Enter a parse tree produced by ProgramParser#mem_access_op.
    def enterMem_access_op(self, ctx:ProgramParser.Mem_access_opContext):
        pass

    # Exit a parse tree produced by ProgramParser#mem_access_op.
    def exitMem_access_op(self, ctx:ProgramParser.Mem_access_opContext):
        pass


    # Enter a parse tree produced by ProgramParser#mem_access_op_single.
    def enterMem_access_op_single(self, ctx:ProgramParser.Mem_access_op_singleContext):
        pass

    # Exit a parse tree produced by ProgramParser#mem_access_op_single.
    def exitMem_access_op_single(self, ctx:ProgramParser.Mem_access_op_singleContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fp.
    def enterInst_fp(self, ctx:ProgramParser.Inst_fpContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fp.
    def exitInst_fp(self, ctx:ProgramParser.Inst_fpContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_f_f_f.
    def enterInst_f_f_f(self, ctx:ProgramParser.Inst_f_f_fContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_f_f_f.
    def exitInst_f_f_f(self, ctx:ProgramParser.Inst_f_f_fContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fcsr.
    def enterInst_fcsr(self, ctx:ProgramParser.Inst_fcsrContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fcsr.
    def exitInst_fcsr(self, ctx:ProgramParser.Inst_fcsrContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_fscsr.
    def enterInst_fscsr(self, ctx:ProgramParser.Inst_fscsrContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_fscsr.
    def exitInst_fscsr(self, ctx:ProgramParser.Inst_fscsrContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_f_x.
    def enterInst_f_x(self, ctx:ProgramParser.Inst_f_xContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_f_x.
    def exitInst_f_x(self, ctx:ProgramParser.Inst_f_xContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_x_f.
    def enterInst_x_f(self, ctx:ProgramParser.Inst_x_fContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_x_f.
    def exitInst_x_f(self, ctx:ProgramParser.Inst_x_fContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_f_f.
    def enterInst_f_f(self, ctx:ProgramParser.Inst_f_fContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_f_f.
    def exitInst_f_f(self, ctx:ProgramParser.Inst_f_fContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_x_f_f.
    def enterInst_x_f_f(self, ctx:ProgramParser.Inst_x_f_fContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_x_f_f.
    def exitInst_x_f_f(self, ctx:ProgramParser.Inst_x_f_fContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_f_ldst.
    def enterInst_f_ldst(self, ctx:ProgramParser.Inst_f_ldstContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_f_ldst.
    def exitInst_f_ldst(self, ctx:ProgramParser.Inst_f_ldstContext):
        pass


    # Enter a parse tree produced by ProgramParser#inst_f_f_f_f.
    def enterInst_f_f_f_f(self, ctx:ProgramParser.Inst_f_f_f_fContext):
        pass

    # Exit a parse tree produced by ProgramParser#inst_f_f_f_f.
    def exitInst_f_f_f_f(self, ctx:ProgramParser.Inst_f_f_f_fContext):
        pass



del ProgramParser