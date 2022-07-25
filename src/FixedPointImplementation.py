# Author: Nimish Shah
# KU Leuven, MICAS

import struct
import sys
import argparse
import random

import logging
logger= logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

#######################################################
#   README
#   
#   Fixed point :
#     FloatingPntToFixedPoint
#     FixedPoint_To_FloatingPoint
#     fix_add
#     fix_mul
#
#   Floating point :
#     flt_to_custom_flt
#     custom_flt_to_flt
#     flt_add
#     flt_add_signed # to be used when dealing with unsigned numbers
#     flt_mul
#
#   Posit
#     set_posit_env
#     flt_to_posit
#     posit_to_flt
#     posit_mul
#     posit_add
#
#
#######################################################


#########################################
## Following functions are useful when doing 
## simulations with custom fixed point representation
########################################
def FloatingPntToFixedPoint(Num_in,INT_bits,FRAC_bits):

    Num=abs(Num_in)

    # if (Num_in>7.999938965):
    if (Num_in>= (2**(INT_bits) - 2**(-FRAC_bits))):  #if we need more than INT_bits to represent the number
        # FinalInt= 131071
        FinalInt=2**(INT_bits+FRAC_bits-1)-1 # considering negative numbers as well
    elif(Num_in<-2**(INT_bits)):
        # FinalInt= 131072
        FinalInt = 2 ** (INT_bits + FRAC_bits-1)
    elif (Num_in==0):
        FinalInt=0
    else:
        #b= str("{0:."+str(FRAC_bits)+"f}".format(Num))
        b= str("{0:.32f}".format(Num))
        INT_part= b.split('.')[0].split()[0]
        INT_part= int(INT_part)
        FRAC_part= Num- INT_part
        Frac_bit_str=''
        Final_Frac_part=0

        for i in range(1,FRAC_bits+1):
            if (FRAC_part<2**(-i)):
                Frac_bit_str= Frac_bit_str+'0'
            else:
                Frac_bit_str= Frac_bit_str+'1'
                Final_Frac_part= Final_Frac_part | 1<<(FRAC_bits-i)

                FRAC_part=FRAC_part- (2**(-i))

        FinalInt= (INT_part<< FRAC_bits) + Final_Frac_part
        if FRAC_part>=2**(-FRAC_bits+1):   ## Perform rounding on FinalInt
            FinalInt= FinalInt+1
            if(FinalInt>=2**(INT_bits+FRAC_bits-1)):    ## After rounding number may exceed the max value
                FinalInt=2**(INT_bits+FRAC_bits-1)-1
    if (Num_in<0):
        FinalInt= -FinalInt
    return FinalInt

def FixedPoint_To_FloatingPoint(FixedPnt_num,INT_DIGITS,FRAC_DIGITS,verb=0): 


    FixedPnt_num_cpy=FixedPnt_num

    if verb:
        print('\nIncoming fixed point number ', FixedPnt_num_cpy)


    if verb:
        print('Comparing ', FixedPnt_num_cpy, ' to ', 2**(INT_DIGITS + FRAC_DIGITS), 'for the sign')
        print(bin(2**(INT_DIGITS + FRAC_DIGITS)))
        print(bin(FixedPnt_num_cpy))

    if (FixedPnt_num & 2**(INT_DIGITS + FRAC_DIGITS)) != 0:  # check last bit  (for sign)
        FixedPnt_num_cpy= -FixedPnt_num_cpy #2's complement


    if verb:
        print('\nFor the integer part ')
        print('Now we shift the input', FRAC_DIGITS, ' places to the right')
        print('From ', bin(FixedPnt_num_cpy), ' to ', bin((FixedPnt_num_cpy >> FRAC_DIGITS)))
        print('And do bitwise and with ', bin(2**(INT_DIGITS+1)-1), ' which results in ', bin((FixedPnt_num_cpy >> FRAC_DIGITS) & 2**(INT_DIGITS+1)-1))
        print((FixedPnt_num_cpy >> FRAC_DIGITS) & 2**(INT_DIGITS+1)-1)

    # Int_part= (FixedPnt_num_cpy >> FRAC_DIGITS) & 0b1111
    Int_part = (FixedPnt_num_cpy >> FRAC_DIGITS) & 2**(INT_DIGITS+1)-1

    if verb:
        print('Int part is ', Int_part, bin(Int_part))


    if verb:
        print('\nNow for the fractional part')
        print('We do and beween ', bin(FixedPnt_num_cpy), ' and ', bin(2**(FRAC_DIGITS)-1))
        print('Resulting in ', bin(FixedPnt_num_cpy & 2**(FRAC_DIGITS)-1 ))

    # Frac_part= FixedPnt_num_cpy & 0x3FFF  #14 bits for fractional
    Frac_part = FixedPnt_num_cpy & 2**(FRAC_DIGITS)-1  # 14 bits for fractional


    Frac_part_in_float=0
    # for i in range(1,15):
    for i in range(1, FRAC_DIGITS+1):
        # if (Frac_part & (1<<(14-i)))!=0:
        if (Frac_part & (1 << (FRAC_DIGITS- i))) != 0:
            Frac_part_in_float= Frac_part_in_float + 2**(-i)

    Final_Float= Int_part + Frac_part_in_float
    if (FixedPnt_num & 2**(INT_DIGITS+FRAC_DIGITS))!=0:
        Final_Float= -Final_Float

    return Final_Float

#######################
## Fixed point adder and multiplier model to mimic exact hardware behaviour
## Arguments passed should be in fixed point format (and not in normal floating point)
#######################
def fix_add(fix_pt_num_1, fix_pt_num_2, INT_L, FRAC_L, verb= 0): # num_1 and num_2 should be in <'INT', 'FRAC'> fromat
  
  assert isinstance(fix_pt_num_1, int)
  assert isinstance(fix_pt_num_2, int)

  res= fix_pt_num_1 + fix_pt_num_2

  # Overflow check
  if (res >= (1 << (INT_L + FRAC_L))):
    res = (1 << (INT_L + FRAC_L)) - 1

  return res

def fix_mul(fix_pt_num_1, fix_pt_num_2, INT_L, FRAC_L, verb= 0): # num_1 and num_2 should be in <'INT', 'FRAC'> fromat
  assert isinstance(fix_pt_num_1, int)
  assert isinstance(fix_pt_num_2, int)

  NUM_L= INT_L + FRAC_L
  res_full= fix_pt_num_1 * fix_pt_num_2
  
  # Round up
  if (res_full & (1 << (FRAC_L-1))):
    res_shifted = (res_full >> FRAC_L) + 1
  else:
    res_shifted = res_full >> FRAC_L
    
  # Overflow Check
  if (res_shifted >= (1 << (INT_L + FRAC_L))):
    res_shifted= (1 << (INT_L + FRAC_L)) - 1
  
  return res_shifted

#########################################
## Following functions are useful when doing 
## simulations with custom floating point representation
########################################
def flt_to_IEEE_str(num):
  # python 2 version:
  # return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

  string= "".join(map(chr, struct.pack('!f', num)))
  return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in string)

def flt_to_IEEE_double_str(num):
  # python 2 version:
  # return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', num))

  string= "".join(map(chr, struct.pack('!d', num)))
  return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in string)

def flt_to_custom_flt(num, EXP_LEN, MNT_LEN, denorm, verb=0): # returns an integer of the form- <SIGN, 'EXP in binary' , 'MNT in binary'>
  
  # assert num >= 0
  #y= flt_to_IEEE_str(num)
  #exp_bin= y[1:(32-23)]
  #mnt= y[9:]
  #exp_bias = (1 << (7)) - 1
  
  y= flt_to_IEEE_double_str(num)
  sign= y[0]
  if (verb): 
    print(y, len(y))

  exp_bin= y[1:(64-52)]
  mnt= y[12:]
  exp_bias = (1 << (10)) - 1

  if (verb):
    print(exp_bin, mnt)
  
  mnt= mnt[:MNT_LEN]
  mnt= int(mnt,2)

  exp_unb = int(exp_bin,2)-exp_bias 
  if (verb==1):
    print('exp_unb, mnt- ', exp_unb, mnt)
  
  ovrflw=0
  undrflw=0
  if (exp_unb > 2**(EXP_LEN-1)):
    if (exp_unb>0):
      ovrflw=1
      exp_unb= 2**(EXP_LEN-1) # 128 for 8 bit exp
      mnt= 2**(MNT_LEN)-1
      if (verb==1):
        print('overflow', exp_unb, mnt)
  
  if (denorm==0):
    if (exp_unb < -(2**(EXP_LEN-1)-1)): # Under flow
      undrflw=1
      exp_unb= -(2**(EXP_LEN-1)-1) # -127 for 8 bit exp
      mnt= 0
      if (verb==1):
        print('underflow', exp_unb, mnt)
  else:
    EXP_LOW_DENORM = -(2**(EXP_LEN-1)-2)
    if (exp_unb < EXP_LOW_DENORM): # Denormalized
      if (verb):
        print('Denormalized')  
      mnt = (1<< MNT_LEN) | mnt
      mnt = mnt >> (EXP_LOW_DENORM-exp_unb)
      exp_unb = -(2**(EXP_LEN-1)-1)
  
    
  exp_fin= exp_unb + 2**(EXP_LEN-1)-1
  
  if (verb==1):
    print(exp_fin, mnt, str(sign + format(exp_fin, '0'+str(EXP_LEN)+'b')) + str(format(mnt, '0'+str(MNT_LEN)+'b')))

  return int(sign + str(format(exp_fin, '0'+str(EXP_LEN)+'b')) + str(format(mnt, '0'+str(MNT_LEN)+'b')), 2)

def custom_flt_to_flt(num, EXP_LEN, MNT_LEN, denorm, verb=0): # returns a floating point number from the <SIGN, 'EXP in binary' , 'MNT in binary'> representation
  # Special case of zero
  if(num == 0): ## If all bits of mant and exp are 0, treat it as zero instead of 1.0 * 2^(-EXP_MIN)
    return 0

  bin_num= str(format(num,'0' + str(1+EXP_LEN+MNT_LEN) +'b'))
  sign= bin_num[0]
  sign= (-1)**int(sign)
  exp_str= bin_num[1:EXP_LEN+1]
  mnt_str= bin_num[EXP_LEN + 1:]
  if verb:
    print("exp_str-", exp_str, "mnt_str-", mnt_str)
  
  exp= int(exp_str,2)- (2**(EXP_LEN-1)-1) # remove bias from exponent
  
  if (denorm == 0):
    mnt= int('1' + mnt_str,2)
  else:
    if (int(exp_str,2) == 0):
      mnt = int('0' + mnt_str,2)
      exp = exp + 1
    else:
      mnt= int('1' + mnt_str,2)

  final_num= float(sign*mnt * (2**(-MNT_LEN + exp)))
  
  if verb:
    print('exp, mnt, final res: ', exp, mnt, final_num)
  return final_num

#######################
## Floating point models for adder and multiplier to mimic exact hardware behaviour
#######################
def flt_add_signed (num_1, num_2, EXP_L, MNT_L, denorm, verb=0): # Model for floating-point addition working on custom_float numbers of form <SIGN, 'EXP in binary' , 'MNT in binary'>
  assert isinstance(num_1, int)
  assert isinstance(num_2, int)
  
  flt_num_1= custom_flt_to_flt(num_1, EXP_L, MNT_L, denorm)
  flt_num_2= custom_flt_to_flt(num_2, EXP_L, MNT_L, denorm)
  
  flt_out= flt_num_1 + flt_num_2

  logger.info(f"flt_num_1, flt_num_2, flt_out: {flt_num_1}, {flt_num_2}, {flt_out}")
  
  out= flt_to_custom_flt(flt_out, EXP_L, MNT_L, denorm)

  return out
  
#######################
## FLOAT ADDER
## Pass 'denorm=1' for denormalized floating numbers
#######################
def flt_add (num_1, num_2, EXP_L, MNT_L, denorm, verb=0): # Model for floating-point addition working on custom_float numbers of form <SIGN, 'EXP in binary' , 'MNT in binary'>

  assert isinstance(num_1, int)
  assert isinstance(num_2, int)
  
  ### Special case for 0 ###
  # if any of the number is zero return the other number
  if (num_1==0):
    return num_2
  if (num_2==0):
    return num_1
    
  # Extracting the mnt and exp part from numbers
  man1= num_1 & ((1 << MNT_L) - 1)
  man2= num_2 & ((1 << MNT_L) - 1)
  exp1= (num_1 >> MNT_L) & ((1 << EXP_L) - 1)
  exp2= (num_2 >> MNT_L) & ((1 << EXP_L) - 1)

  sign1= (num_1 >> (MNT_L + EXP_L)) & 1
  sign2= (num_2 >> (MNT_L + EXP_L)) & 1
  sign1 = (-1) ** sign1
  sign2 = (-1) ** sign2
  
  # Adding the implicit 1 explicitly
  if (denorm == 0):
    man1E= (1 << MNT_L) | man1 
    man2E= (1 << MNT_L) | man2 
  else:
    if (exp1 > 0):
      man1E= (1 << MNT_L) | man1 
    else:
      man1E= man1
    if (exp2 > 0):
      man2E= (1 << MNT_L) | man2 
    else:
      man2E= man2
  
  # Exponent Difference Calculation
  exp1S= exp1
  exp2S= -exp2

  dExpS= exp1S + exp2S

  if (dExpS<0):
    dExp= -dExpS
  else:
    dExp= dExpS

  if (denorm == 0):
    shiftExp= dExp
  else:
    if ((exp1==0) != (exp2==0)): # One of the number is denormal and other is normal
      shiftExp= dExp - 1
    else:
      shiftExp= dExp

  # Mantissa Calculation
  if (dExpS<0):
    manPreShift = man1E << 1 # Extra bit is padded. This would be required for rounding
    manNormal = man2E << 1
    signShift= sign1
    signNormal= sign2
  else:
    manPreShift = man2E << 1
    manNormal = man1E << 1
    signShift= sign2
    signNormal= sign1

  manShift= manPreShift >> shiftExp
  
  man_raw=  manShift + manNormal
  # man_raw= (signShift * manShift) + (signNormal*manNormal)
  # if man_raw < 0:
  #   sign_out= 1
  #   man_raw = -man_raw
  # else:
  #   sign_out = 0

  if (man_raw & (1 << (MNT_L + 2))): # if the sum of two mantissa's has an overflow then we have to slice man_raw differntly
    manPreRound = man_raw >> 1 
  else:
    manPreRound = man_raw
  
  if (manPreRound & 1): # Rounding
    manRound= (manPreRound >> 1) + 1
  else:
    manRound= (manPreRound >> 1)
  
  if (manRound & (1 << (MNT_L + 1))): # if overflow occurs due to round
    manPreOverflow = manRound >> 1
  else:
    manPreOverflow = manRound

  if (exp1 < exp2):
    expSel= exp2
  else:
    expSel= exp1
  
  expInc = expSel + 1
  if (denorm == 0):
    increment= ((man_raw & (1 << (MNT_L + 2))) or (manRound & (1<< (MNT_L + 1))))
  else:
    increment= (expSel==0 and (man_raw & (1 << (MNT_L + 1)))) or (expSel!=0 and ((man_raw & (1 << (MNT_L + 2))) or (manRound & (1<< (MNT_L + 1)))))
  
  if (increment):
    expPreOverflow = expInc;
  else:
    expPreOverflow = expSel

  # Exp Overflow detection
  if (verb):
    print((expPreOverflow & (1 << EXP_L)))
  if (expPreOverflow & (1 << EXP_L)):
    exp = (1 << EXP_L) - 1 # All 1s
    man = (1 << MNT_L) - 1 
    logger.info("flt add overflow")
  else:
    exp = expPreOverflow
    man = manPreOverflow & ((1 << MNT_L) - 1)
  
  if (verb== 1):
    print("exp1, exp2, man1E, man2E- ",exp1, exp2, man1E, man2E)
    print("increment, (man_raw & (1 << (MNT_L + 2))), (manRound & (1<< (MNT_L + 1))) -",increment, (man_raw & (1 << (MNT_L + 2))), (manRound & (1<< (MNT_L + 1)))) 
    print("man1E, man2E, manShift, manNormal- ", bin(man1E), bin(man2E), bin(manShift), bin(manNormal))  
    print("man_raw, manPreRound, manRound, manPreOverflow- ", bin(man_raw), bin(manPreRound), bin(manRound), bin(manPreOverflow))
    print("exp, man- ", exp, man, expInc, expSel, expPreOverflow)
  
  return (exp << MNT_L) | man 


#######################
## FLOAT MULTIPLIER  ##
## Pass denorm=1 for denormalized numbers
#######################
def flt_mul(flt_num_1, flt_num_2, EXP_L, MNT_L, denorm, verb=0):
  assert isinstance(flt_num_1, int)
  assert isinstance(flt_num_2, int)

  ### Special case for 0 ###
  # if any of the number is zero return result as zero
  if (flt_num_1==0 or flt_num_2==0):
    return 0

  # Extracting the mnt and exp part from numbers
  sigX= flt_num_1 & ((1 << MNT_L) - 1)
  sigY= flt_num_2 & ((1 << MNT_L) - 1)
  expX= (flt_num_1 >> MNT_L) & ((1 << EXP_L) - 1)
  expY= (flt_num_2 >> MNT_L) & ((1 << EXP_L) - 1)
  sign1= (flt_num_1 >> (MNT_L + EXP_L)) & 1
  sign2= (flt_num_2 >> (MNT_L + EXP_L)) & 1

  sign1 = (-1) ** sign1
  sign2 = (-1) ** sign2

  # Adding the implicit 1 explicitly
  if (denorm == 0):
    sigX= (1 << MNT_L) | sigX 
    sigY= (1 << MNT_L) | sigY 
  else:
    if (expX > 0):
      sigX= (1 << MNT_L) | sigX 
    if (expY > 0):
      sigY= (1 << MNT_L) | sigY 

  ### NORMALIZED MULTIPLICATION ###
  if (denorm == 0):
    sigProd= sigX * sigY
    norm = 0
    if (sigProd & (1 << (2*MNT_L+1))):
      norm = 1
    else:
      norm = 0

    bias= (1 << (EXP_L-1)) - 1
    expPostNorm= expX + expY - bias + norm

    if (norm):
      sigProdExt_pre= (sigProd & ((1 << (2*MNT_L+1)) - 1)) << 1
    else:
      sigProdExt_pre= (sigProd & ((1 << (2*MNT_L)) - 1)) << 2

    if (expPostNorm < 0): # Underflow
      expPostNorm_fin = 0
      sigProdExt= 0
      if (verb):
        print("Underflow")
      logger.info("flt mul underflow")
    elif (expPostNorm & (1<<EXP_L)): # Overflow
      expPostNorm_fin= (1 << (EXP_L+1)) - 1
      sigProdExt= ((1 << MNT_L) - 1) << (MNT_L+2)
      if (verb):
        print("Overflow")
      logger.info("flt mul overflow")
    else:
      expPostNorm_fin = expPostNorm
      sigProdExt= sigProdExt_pre

    expSig = (expPostNorm_fin << MNT_L) | ((sigProdExt >> (MNT_L + 2)) & ((1 << MNT_L) - 1))
    sticky = 0
    
    if (expSig & ((1<<MNT_L)-1) != (1<<MNT_L)-1): # All bits of Mantisaa are not 1
      if (sigProdExt & (1 << (MNT_L+1))):
        sticky = 1
    else:
      sticky = 0
    
    expSigPostRound = sticky + expSig
    out = expSigPostRound & ((1 << (EXP_L + MNT_L)) - 1)

    sign_out= sign1 * sign2
    if sign_out == -1:
      sign_out= 1 << (EXP_L + MNT_L)
    elif sign_out == 1:
      sign_out= 0
    else:
      assert 0
    
    out |= sign_out

    return out
  
  ### DENORMAL MULTIPLICATION #####
  else: # if(denorm==0) 
    if (expX == 0):
      expX = 1
    if (expY == 0):
      expY = 1

    sigProd= sigX * sigY

    pos_one_in_sigProd = lead_1_pos(sigProd, MNT_L)
    
    bias= (1 << (EXP_L-1)) - 1
    expSum= expX + expY - bias
    
    left_shift= 2*MNT_L - pos_one_in_sigProd

    expPostNorm= expSum - left_shift
    
    if (verb):
      print("sigX, sigY, expX, expY- ", bin(sigX), bin(sigY), bin(expX), bin(expY))
      print("left_shift, expSum, expPostNorm, bin(sigProd):", left_shift, expSum, expPostNorm, bin(sigProd)) 

    if (expPostNorm >= (1 << EXP_L) ): # Overflow
      if (verb):
        print('Overflow')
      return ((1 << (EXP_L + MNT_L)) - 1) # All ones
      logger.info("flt mul overflow")
    
    elif (expPostNorm > 0): # Output can be normalized
      if (verb):
        print('Output can be normalized')
      
      if (left_shift >= 0):
        sigProd_shifted = sigProd << left_shift # This brings leading 1 at 2*MNT_L position (i.e. (2*MNT_L+1)th bit)
      elif (left_shift < 0):
        sigProd_shifted = sigProd >> abs(left_shift) # This brings leading 1 at 2*MNT_L position (i.e. (2*MNT_L+1)th bit)
      
      sigPreRound = (sigProd_shifted >> MNT_L) & ((1 << MNT_L) - 1)
      
      if (sigPreRound != (1 << MNT_L)-1 ): 
        if (sigProd_shifted & (1<<(MNT_L-1))) : # Check if sticky bit was 1 
          sigRound = sigPreRound + 1  # This rounding will not create overflow
        else:
          sigRound = sigPreRound
      else:
        if (verb):
          print("Not rounding because all bits are 1")
        sigRound = sigPreRound
      
      return (expPostNorm << MNT_L) | sigRound


    elif (expPostNorm <= 0): # Either Denormalizaed o/p or Underflow
      if (expPostNorm <= -(MNT_L)): # Smaller than smallest denormal number, hence underflow
        if (verb):
          print('Underflow')
        return 0
        logger.info("flt mul underflow")

      else: # Representation in DENORMAL number possible
        if (verb):
          print("Denormal output")
        exp_final = 0
        shift = expSum - 1 # expSum would be a negative value or zero and hence left shift would definitely be negative
        if (shift >= 0): # will never hit this
          sigProd_shifted = sigProd << shift 
        else:
          sigProd_shifted = sigProd >> abs(shift)
          
        sigPreRound = (sigProd_shifted >> MNT_L) & ((1 << MNT_L) - 1)
        
        if (sigPreRound != (1 << MNT_L)-1 ): # Check if overflow can happen 
          if (sigProd_shifted & (1 << (MNT_L-1))): 
            sigRound = sigPreRound + 1  
          else:
            sigRound = sigPreRound
        else:
          sigRound = sigPreRound
       
        if (sigRound & (1 << MNT_L)): # Rounding created overflow, so don't round
          return (exp_final << MNT_L) | (sigRound-1)
          #if (verb):
          #  print 'Output could be normalized after rounding'
          #return ((exp_final+1) << MNT_L) | (sigRound & ((1 << MNT_L) - 1))
        else:
          return (exp_final << MNT_L) | sigRound
    

def lead_1_pos(sigProd, MNT_L):
  for i in range(2*MNT_L + 4, -1, -1):
    if (sigProd & (1 << i)):
      return i

def test_flt_add(n_iter, EXP_L, MNT_L):
  EXP_BIAS= 2**(EXP_L-1)
  for loop in range(n_iter):
    for i in range((2**EXP_L)-1):
      for j in range((2**MNT_L)-1):
        
        num_1= (i << MNT_L) | j
        
        rand_mnt= random.randint(0, (2**MNT_L)-1)
        rand_exp= random.randint(0, (2**EXP_L)-1)
        num_2= rand_exp << MNT_L | rand_mnt
        
        res= custom_flt_to_flt(flt_add(num_2 , num_1 , EXP_L, MNT_L), EXP_L, MNT_L)
        
        num_1_flt= custom_flt_to_flt(num_1, EXP_L, MNT_L)
        num_2_flt= custom_flt_to_flt(num_2, EXP_L, MNT_L)
        
        gld_res= num_1_flt + num_2_flt 
        if (flt_to_custom_flt(res, EXP_L, MNT_L) != ((1 << (EXP_L+MNT_L))-1)): # Not overflown
          if (abs(gld_res - res) > flt_add_err_model(num_1, num_2, EXP_L, MNT_L) ):
            print("Error at loop, i,j=", loop, i,j, gld_res, res, num_1_flt, num_2_flt)
            custom_flt_to_flt(flt_add(num_1 , num_2 , EXP_L, MNT_L, 1), EXP_L, MNT_L)
            assert 0, "Model violated"
            exit(1)

def test_flt_mul(n_iter, EXP_L, MNT_L, denorm= False):
  for loop in range(n_iter):
    for exp in range((1 << EXP_L) - 1):
      for mnt in range((1 << MNT_L) - 1):
        
        num_1 = (exp << MNT_L) | mnt
        if (random.randint(0, 100) < 90): 
          rand_exp= 0
        else:
          rand_exp= random.randint(0, (2**EXP_L)-1)
        rand_mnt= random.randint(0, (2**MNT_L)-1)
        
        num_2= rand_exp << MNT_L | rand_mnt
        
        res_cust= flt_mul(num_2 , num_1 , EXP_L, MNT_L, denorm)
        res= custom_flt_to_flt(res_cust, EXP_L, MNT_L, denorm)
        
        num_1_flt= custom_flt_to_flt(num_1, EXP_L, MNT_L, denorm)
        num_2_flt= custom_flt_to_flt(num_2, EXP_L, MNT_L, denorm)
        gld_res= num_1_flt * num_2_flt
        
        if (abs(gld_res - res) > flt_mul_err_model(num_1, num_2, EXP_L, MNT_L, denorm) and res_cust != (1<<(EXP_L+MNT_L))-1):
          print("num_1_exp, num_1_mnt: ", exp, mnt)
          print("num_2_exp, num_2_mnt: ", rand_exp, rand_mnt)
          print(num_1_flt, num_2_flt)
          print(gld_res, res)
          print("Obs. error: ", abs(gld_res -res))
          flt_mul(num_2 , num_1 , EXP_L, MNT_L,denorm , 1)
          flt_mul_err_model(num_1, num_2, EXP_L, MNT_L, denorm, 1)
          print("Error!")
          exit(1)
       
        res_2_cust= flt_mul(res_cust, num_2, EXP_L, MNT_L, denorm)
        res_2= custom_flt_to_flt(res_2_cust, EXP_L, MNT_L, denorm)
        gld_res= res * num_2_flt
        if (abs(gld_res - res_2) > flt_mul_err_model(res_cust, num_2, EXP_L, MNT_L, denorm) and res_2_cust != (1<<(EXP_L+MNT_L))-1):
          print('Error')
          print(res, num_2_flt, res_2, gld_res)
          flt_mul_err_model(res_cust, num_2, EXP_L, MNT_L, denorm, 1)
          assert 0, "Model violated"
          exit(1)
         
      
####################
## Error model for floating adder:
## Absolute value of the error of the float adder adding two numbers 
## cannot exceed the value given by the model
####################
def flt_add_err_model(num_1, num_2, EXP_L, MNT_L, verb=0): 
  # Extracting the mnt and exp part from numbers
  man1= num_1 & ((1 << MNT_L) - 1)
  man2= num_2 & ((1 << MNT_L) - 1)
  exp1= (num_1 >> MNT_L) & ((1 << EXP_L) - 1)
  exp2= (num_2 >> MNT_L) & ((1 << EXP_L) - 1)
  
  EXP_BIAS= 2**(EXP_L-1)-1

  abs_error= 2**(-MNT_L+1) *  (2**(max(exp1,exp2)-EXP_BIAS + 1))
  
  return abs_error


####################
## Error model for floating multiplier:
## Absolute value of the error of the float multiplier adding two numbers 
## cannot exceed the value given by the model
####################
def flt_mul_err_model(num_1, num_2, EXP_L, MNT_L, denorm, verb=0): 
  # Extracting the mnt and exp part from numbers
  man1= num_1 & ((1 << MNT_L) - 1)
  man2= num_2 & ((1 << MNT_L) - 1)
  exp1= (num_1 >> MNT_L) & ((1 << EXP_L) - 1)
  exp2= (num_2 >> MNT_L) & ((1 << EXP_L) - 1)

  EXP_BIAS= 2**(EXP_L-1) - 1

  if (verb):
    print(exp1, exp2, EXP_BIAS, EXP_L)
  if (exp1+ exp2 - EXP_BIAS > ((1 << EXP_L) - 1) ): # Overflow
    abs_error = (2 ** (exp1+ exp2 - EXP_BIAS + 1)) - (2 ** (EXP_L + 1))
    if (verb):
      print("Overflow in flt_mul_err_model")
  elif (exp1+ exp2 - EXP_BIAS < denorm ): # Underflow
    if (denorm==0):
      abs_error = 2**(-EXP_BIAS)
    else:
      abs_error = 2**(-EXP_BIAS+1-MNT_L)
    if (verb):
      print("Underflow in flt_mul_err_model")
  else:
    abs_error = 2** (exp1 + exp2 - EXP_BIAS - MNT_L)

  if (verb):
    print("Abs. error: ", abs_error)

  return abs_error


def run(args):
  test = 1
  
  print(FixedPoint_To_FloatingPoint(13370896, 6, 26))
  print(custom_flt_to_flt(548227147, 8, 23, 0))
  if (test):
    test_flt_add(1)
    EXP_L= 4
    MNT_L= 10
    num_1= flt_to_custom_flt(1.125, EXP_L, MNT_L,1)
    num_2= flt_to_custom_flt(0.175, EXP_L, MNT_L,1)
    res_1= flt_add(num_1,num_2, EXP_L, MNT_L, 1)
    res_1= custom_flt_to_flt(res_1, EXP_L, MNT_L, 1)
  
    num_2= flt_to_custom_flt(0.125, EXP_L, MNT_L, 0)
    res_2= custom_flt_to_flt(num_2, EXP_L, MNT_L, 0)
    print(res_1, res_2)
    
    num_1_flt= 2**(-(2**(EXP_L-1))-MNT_L+4)
    num_2_flt= 2**(-4)
    num_1= flt_to_custom_flt(num_1_flt, EXP_L, MNT_L,1)
    num_2= flt_to_custom_flt(num_2_flt, EXP_L, MNT_L,1)
    res_1= flt_mul(num_1, num_2, EXP_L, MNT_L,1,1)
    res_1= custom_flt_to_flt(res_1, EXP_L, MNT_L, 1)
    print(res_1, num_1_flt, num_2_flt)
    exit(0)

    num_in1=2.73485
    num_in2=3.743878
    
    int_digit=4
    frac_digits=20

    fixed_num1=FloatingPntToFixedPoint(num_in1,int_digit,frac_digits)

    fixed_num2 = FloatingPntToFixedPoint(num_in2, int_digit, frac_digits)


    print('\nFloating input 1: ', num_in1)
    print('Floating input 2: ', num_in2)

    print('\nFixed input 1: ', fixed_num1)
    print('Fixed input 2: ', fixed_num2)

    print('\nFixed input 1 to float: ',FixedPoint_To_FloatingPoint(fixed_num1,int_digit,frac_digits,0))
    print('\nFixed input 2 to float: ', FixedPoint_To_FloatingPoint(fixed_num2,int_digit,frac_digits,0))

    print('\nFixed input 1 bin: ', bin(fixed_num1))
    print('Fixed input 2 bin: ', bin(fixed_num2))

    print('\nFP addition ', fixed_num1+fixed_num2)
    print('\nFP addition in bin', bin(fixed_num1+fixed_num2))

    print('\nFP addition to float', FixedPoint_To_FloatingPoint(fixed_num1+fixed_num2,int_digit,frac_digits,[]))
    print('Actual float add:', num_in1 + num_in2)
    # print '\nFP addition in bin to float', bin(FixedPoint_To_FloatingPoint(fixed_num1+fixed_num2,int_digit,frac_digits))

    print('\nFP mult ', fixed_num1*fixed_num2)
    print('\nFP mult in bin', bin(fixed_num1*fixed_num2))


    res=FixedPoint_To_FloatingPoint(fixed_num1*fixed_num2>>frac_digits,int_digit,frac_digits,[])
    print('\nFP mult to float', res)
    print('Actual float mult:', num_in1 * num_in2)



def main(argv=None):
    parser = argparse.ArgumentParser(description='Example Python Code')
    args = parser.parse_args(argv)

    run(args)


if __name__ == "__main__":
    sys.exit(main())
