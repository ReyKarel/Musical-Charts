#!/usr/bin/env python
# coding: utf-8

# In[18]:


'''
TO DO:
add scale change functionality
    add scale changing logic
add new composing methods

solve issue of different methods taking more or less bars




~~~~~~~~~~~~~~~~~~~~~~~


'''


def change_scale(scale):   # Run this bad boy in chunks over the DF
    newScale = scale_range(range(scale[0]+7,120))
    return newScale


# In[19]:


from midiutil import MIDIFile


# In[20]:


import pandas as pd
import yfinance as yf
import numpy as np
from scipy import stats
import statistics
from random import choice, choices, triangular, shuffle


# In[ ]:




# In[28]:


def fit_to_scale(data, scale, round_down = False):
    scaled = []
    for i in data:
        if i in scale:
            scaled.append(i)
        elif (i + 1) in scale:
            scaled.append(i+1)
        else:
            scaled.append(i-1)

    return scaled


# In[16]:


def humanize(std=0.012):
    return round(np.random.normal(0,std),3)

# In[24]:


def scale_range(notes, minor=False, harmonic=False):
    if minor == False:
        scale = [2,2,1,2,2,2,1]
    else:
        scale = [2,1,2,2,1,2,2]
        if harmonic:
            scale = [2,1,2,2,1,3,1]
    scaled = []
    i=0 #counter for entire note range
    x=0 #counter for looping over the sale steps
    while i < len(notes):
        scaled.append(notes[i])
        if x > (len(scale) - 1):
            x = 0
        i += scale[x]
        x += 1
    return scaled
        
C_maj = scale_range(range(0,108))
print(C_maj)


# In[ ]:





# In[25]:


[9]

# In[27]:


def normalize(list_normal, mult=1, start=0):
    max_value = max(list_normal)
    min_value = min(list_normal)
    for i in range(len(list_normal)):
        if list_normal[i] == None:
            list_normal[i] = list_normal[i-1]
        list_normal[i] = int((((list_normal[i] - min_value) / (max_value - min_value)) * mult) + start)
    return list_normal

# In[26]:


#function to treat the ohlc
#before rounding check for red/green candles and % distance of wicks
#normalize volume to between 80-100


#function to treat extreme volume


# def zscore(df):
#     z_scores = stats.zscore(df)
#     abs_z_scores = np.abs(z_scores)
#     filtered_entries = (abs_z_scores < 3).all(axis=1)
#     removed = len(df) - len(filtered_entries)
#     for i in df:
#         if i not in filtered_entries:
#             i = filtered_entries.max()
#     #new_df = df[filtered_entries]
#     return df    

def his_los(df):
    for i in range(len(df)):
        df.Low[i] = min((df.Open[i],df.Close[i])) - df.Low[i]
        df.High[i] = df.High[i] - max((df.Open[i],df.Close[i]))
    normalize(df.High,4,1)
    normalize(df.Low,4,1)
    for i in range(len(df)):
        df.Low[i] = int(df.Low[i])
        df.High[i] = int(df.High[i])
    for i in ('High','Low'):
        df[i] = df[i].astype(int)
    
def norm_dir(col):
    for i in col:
        i = abs(i)
    return normalize(col,10)

 
def treat_chart(chart, scale, chunks = 4):
    
    
    if 'Adj Close' in chart.columns:
            chart.Close = chart['Adj Close']
            chart.drop(columns='Adj Close',inplace=True)
#     chart['harmony'] = ""
    chart['scale'] = ''
        
    high_low = chart['High'] - chart['Low']
    high_close = np.abs(chart['High'] - chart['Close'].shift())
    low_close = np.abs(chart['Low'] - chart['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    chart['atr'] = true_range.rolling(5).sum()/5
    chart.drop(chart.head(5).index,inplace=True)
    chart.drop(chart.tail(1).index,inplace=True)
    chart.drop(chart.tail(len(chart)%chunks).index,inplace=True)
    part = int(len(chart)/chunks)
    for i in range(len(chart)):
        if i % part == 0 and i != 0:
            scale = change_scale(scale)
            chart.scale[i] = scale
        else:
            chart.scale[i] = scale
    
    
    
    chart['DC top'] = (chart.High.rolling(window=20).max())
    chart['DC bottom'] = (chart.Low.rolling(window=20).min())
    chart['DC top'][:20] = chart.High[:20].max()
    chart['DC bottom'][:20] = chart.Low[:20].min()
    chart['new low'] = (chart['DC bottom'].diff()) < 0
    chart['new high'] = chart['DC top'].diff() > 0
    chart['new low'][0] = False
    chart['new high'][0] = False
    
    
    
    a = 0
    b = 1
#     for i in range(chunks):     
#         chart['scale'][a*part:b*part] = scale
#         scale = change_scale(scale)
        
#         a += 1
#         b += 1
    
    chart['up'] = chart.Open < chart.Close
    his_los(chart)
    chart['harmony'] = chart.Close - chart.Open
    chart['harmony'] = normalize(chart['harmony'], 10, 1)
    for column in ('Open', 'Close'):
        chart[column] = normalize(chart[column],12, 60).round().astype(int)
    
#     chart.Open = fit_to_scale(chart.Open.astype(int), chart.scale)
    
    chart.harmony[chart.up == False] *= -1
    chart.harmony = chart.harmony.astype(int)
    
    chart.Close = chart.Open + chart.harmony
    chart.High[chart.harmony > 0] = (chart.Close + chart.High)
    chart.High[chart.harmony < 0] = (chart.Open + chart.High)
    chart.Low[chart.harmony > 0] = (chart.Open - chart.Low)
    chart.Low[chart.harmony < 0] = (chart.Close - chart.Low)
#     for i in (chart.Low,chart.High,chart.Open,chart.Close):
#         i = np.array(i,dtype=int)
    for i in range(len(chart)):
        chart.Low[i] = fit_to_scale([chart.Low[i],5], chart.scale[i])[0]
        chart.High[i] = fit_to_scale([chart.High[i],5], chart.scale[i])[0]
        chart.Open[i] = fit_to_scale([chart.Open[i]], chart.scale[i])[0]
        chart.Close[i] = fit_to_scale([chart.Close[i],5], chart.scale[i])[0]
#     chart['harmony'] = chart.Close - chart.Open

    a = 0
    b = 1
    for i in range(chunks):     
        chart.Volume[a*part:b*part] = normalize(chart.Volume[a*part:b*part],20,72)
        
        a += 1
        b += 1
    chart['DC top'] = (chart.High.rolling(window=20).max())
    chart['DC bottom'] = (chart.Low.rolling(window=20).min())
    chart['DC top'][:20] = chart.High[:20].max()
    chart['DC bottom'][:20] = chart.Low[:20].min()
    for i in ['DC top', 'DC bottom']:
        chart[i] = chart[i].astype(int)

        chart['melody'] = round((chart['DC top'] + chart['DC bottom']) / 2)
    chart['atr'] = normalize(chart.atr, 10).astype(int)
    
    chart.melody[chart['harmony']<0] = (chart['DC top'] + chart.melody)/2
    chart.melody[chart['harmony']>0] = ((chart['DC bottom'] + chart.melody)/2).astype(int)
    chart.melody[chart['new low']] = chart['DC bottom']
    chart.melody[chart['new high']] = chart['DC top']
    chart.melody = chart.melody.astype(int)
    for i in range(len(chart)):
        chart.melody[i] = fit_to_scale([chart.melody[i],5], chart.scale[i])[0]

    chart['candle'] = chart.High / chart.Low
    chart.candle = normalize(chart.candle, 99).astype(int)
    return chart


# In[21]:


qf = yf.download('TSLA',period='6mo', interval='1d')
print(qf)
treat_chart(qf, C_maj)
print(qf)



# In[175]:


qf[qf['new high']]


# In[101]:



# In[ ]:








# In[29]:


def harmonize(seq,scale,strength):
    '''Function takes a sequence of notes and adds harmony according to given strength'''

    # if abs(strength) > 8:
    #     if strength > 0:
    #         return fit_to_scale([x+4 for x in reversed(seq)],scale)
    #     else:
    #         return fit_to_scale([x-4 for x in reversed(seq)],scale)
    # elif abs(strength) > 4:
    #     if strength > 0:
    #         return fit_to_scale([x+7 for x in reversed(seq)],scale)
    #     else:
    #         return fit_to_scale([x-7 for x in reversed(seq)],scale)
    # else:
    #     if strength > 0:
    #         return fit_to_scale([x+11 for x in reversed(seq)],scale)
    #     else:
    #         return fit_to_scale([x-11 for x in reversed(seq)],scale)
    
    boollist = [strength > 8, strength > 4, strength > 0, strength < -8,  strength < -4, strength < 0]
    i = boollist.index(True)
    hlist = [11, 7, 4, -11, -7, -4]
    h = hlist[i]
    return fit_to_scale([x+h for x in reversed(seq)],scale)


# In[30]:


def arpeggio(note, chart, pos, scale, strength):
    '''Function takes an index of a row from the treated chart and a 1-3 level of strength
    and returns an arpeggio according to market direction and strength'''
    if abs(strength) > 8:
        strength = 3
    elif abs(strength) > 4:
        strength = 2
    else:
        strength = 1
    vol_up = [chart.Volume[pos]*1.05,chart.Volume[pos]*0.95,chart.Volume[pos],chart.Volume[pos]*0.95]
    vol_down = [chart.Volume[pos],chart.Volume[pos]*1.05,chart.Volume[pos],chart.Volume[pos]*0.95]
    if chart.up[pos]:
#         return fit_to_scale([note, note+5,note+3,note+7], scale), vol_up
        if strength == 1:
            return fit_to_scale([chart.Open[pos],chart.Low[pos]-2, chart.Close[pos],chart.High[pos]+2], scale), vol_up
        elif strength == 2:
            return fit_to_scale([chart.Open[pos],chart.Low[pos]-4, chart.Close[pos],chart.High[pos]+4],scale), vol_up
        else:
            return fit_to_scale([chart.Open[pos],chart.Low[pos]-6, chart.Close[pos],chart.High[pos]+6],scale), vol_up
    else:
#         return fit_to_scale([note+5, note+3,note,note-3], scale), vol_down
        if strength == 1:
            return fit_to_scale([chart.Open[pos],chart.High[pos]+2,chart.Close[pos],chart.Low[pos]-2], scale), vol_down
        elif strength == 2:
            return fit_to_scale([chart.Open[pos],chart.High[pos]+4,chart.Close[pos],chart.Low[pos]-4],scale), vol_down
        else:
            return fit_to_scale([chart.Open[pos],chart.High[pos]+6,chart.Close[pos],chart.Low[pos]-6],scale), vol_down


# In[31]:


# def arpeggio(chart, pos, scale, strength):
#     '''Function takes an index of a row from the treated chart and a 1-3 level of strength
#     and returns an arpeggio according to market direction and strength'''
#     if abs(strength) > 8:
#         strength = 3
#     elif abs(strength) > 4:
#         strength = 2
#     else:
#         strength = 1
#     vol_up = [chart.Volume[pos]*1.05,chart.Volume[pos]*0.95,chart.Volume[pos],chart.Volume[pos]*0.95]
#     vol_down = [chart.Volume[pos],chart.Volume[pos]*1.05,chart.Volume[pos],chart.Volume[pos]*0.95]
#     if chart.up[pos]:
#         return fit_to_scale([chart.Open[pos], chart.Open[pos]+5,chart.Open[pos]+3,chart.Open[pos]+7], scale), vol_up
#         if strength == 1:
#             return fit_to_scale([chart.Open[pos],chart.Low[pos]-2, chart.Close[pos],chart.High[pos]+2], scale), vol_up
#         elif strength == 2:
#             return fit_to_scale([chart.Open[pos],chart.Low[pos]-4, chart.Close[pos],chart.High[pos]+4],scale), vol_up
#         else:
#             return fit_to_scale([chart.Open[pos],chart.Low[pos]-6, chart.Close[pos],chart.High[pos]+6],scale), vol_up
#     else:
#         return fit_to_scale([chart.Open[pos]+5, chart.Open[pos]+3,chart.Open[pos],chart.Open[pos]-3], scale), vol_down
#         if strength == 1:
#             return fit_to_scale([chart.Open[pos],chart.High[pos]+2,chart.Close[pos],chart.Low[pos]-2], scale), vol_down
#         elif strength == 2:
#             return fit_to_scale([chart.Open[pos],chart.High[pos]+4,chart.Close[pos],chart.Low[pos]-4],scale), vol_down
#         else:
#             return fit_to_scale([chart.Open[pos],chart.High[pos]+6,chart.Close[pos],chart.Low[pos]-6],scale), vol_down


class composition:
    def __init__(self) -> None:
        self.arp = []
        self.vols = []
        self.harmo = []
        self.harmoVols = []
        self.bassline = []
        self.melody = []
        self.durations = []
        self.chrds = []
    
    def assign(self,arp, vols, harmo, harmoVols, melody, durations, chrds):
        self.arp.extend(arp)
        self.vols.extend(vols)
        self.harmo.extend(harmo)   
        self.harmoVols.extend(harmoVols)
        self.melody.extend(melody)
        self.durations.extend(durations)
        self.chrds.extend(chrds)

def compose(chart, scale):
    o,h,l,c,v,harmony = chart.Open, chart.High, chart.Low, chart.Close, chart.Volume, chart.harmony
    arp = []
    vols = []
    harmo = []
    harmoVols = []
    bassline = []
    melody = []
    durations = []
    chrds = []
    scl = chart.scale
    for i in range(len(chart)):
        cand = (chart.candle[i] // 25) + 1
        step = chart.atr[i] // 2
        dct = int(chart['DC top'][i])
        dcb = int(chart['DC bottom'][i])
        high = scl[i].index(h[i])               
        low = scl[i].index(l[i])
        note = scl[i].index(chart['melody'][i])
        if i > 0:
            op = c[i-1]
        else:
            op = o[0]
        if o[i] % 12 <= 6:
            bass = int(48 + (o[i] % 12))
        else:
            bass = int(48 - (o[i] % 12))
            
        if  abs(chart['harmony'][i]) < 3:
            arp.append(chord(op,scl[i],harmony[i]))
#             arp.extend(arpeggio(op,chart,i,scale,harmony[i])[0])
            vols.extend([75])
            harmo.append(71)
            harmoVols.extend([v[i]-15])
            melody.extend([scl[i][note]])
            durations.extend([1])
            chrds.extend([True])
            
        elif chart['harmony'][i] < 6:
            arp.extend(glide(chart.Low[i if i == 0 else i-1],chart.High[i],scl[i])[:cand])
            vols.extend([v[i]]*cand)
            harmo.extend(harmonize(arp[-cand:],scl[i],harmony[i])[:cand])   
            harmoVols.extend([v[i]-15]*cand)
            # melody.extend([scl[i][note],scl[i][note]+step,scl[i][note]+2+step,scl[i][note]+step][:cand])
            durations.extend([1/cand]*cand)
            chrds.extend([False]*cand)

        else:
            
            arp.extend(arpeggio(op,chart,i,scl[i],harmony[i])[0][:cand])
            vols.extend([v[i]]*cand)
            harmo.extend(harmonize(arp[-cand:],scl[i],harmony[i])[:cand])
            harmoVols.extend([v[i]-15]*cand)
            # melody.extend([scl[i][note],scl[i][note]+step,scl[i][note]+2+step,scl[i][note]+step][:cand])
            melody.extend([scl[i][note]])
            durations.extend([1/cand]*cand)
            chrds.extend([False]*cand)

#         if harmony[i] >= 0:
#             if harmony[i] < 6:
#                 arp.extend(arpeggio(chart,i,))
#                 vols.extend([v[i]]*4)
#                 harmo.extend([op-7, scale[low - 2], scale[high + 2],op])
#                 harmoVols.extend([v[i]-15]*4)
#                 bassline.extend([bass, bass])
#                 melody.extend([scale[note],scale[note]+step,scale[note]+2+step,scale[note]+step])
#             elif harmony[i] < 8:
#                 arp.extend(traverse(op,c[i],scale))
#                 vols.extend([v[i]]*4)
#                 harmo.extend([op-7, scale[low - 2], scale[high + 2],op])
#                 harmoVols.extend([v[i]-15]*4)
#                 bassline.extend([bass, bass])
#                 melody.extend([scale[note],scale[note]+step,scale[note]+2+step,scale[note]+step])
#             else:
#                 arp.extend(traverse(op,c[i],scale))
#                 vols.extend([v[i]]*4)
#                 harmo.extend([op-7, scale[low - 2], scale[high + 2],op])
#                 harmoVols.extend([v[i]-15]*4)
#                 bassline.extend([bass, bass])
#                 melody.extend([scale[note],scale[note]+step,scale[note]+2+step,scale[note]+step])
#         else:
#             if harmony[i] > -6:
#                 arp.extend(traverse(op,c[i],scale))
#                 vols.extend([v[i]]*4)
#                 harmo.extend([op-7, scale[high + max((abs(harmony[i]) // 2),2)],op, scale[low - max((abs(harmony[i]) // 2),2)]])
#                 harmoVols.extend([v[i]-15]*4)
#                 bassline.extend([bass, bass])
#                 melody.extend([scale[note],scale[note]-step,scale[note]-2-step,scale[note]-step])
#             elif harmony[i] > -8:
#                 arp.extend(traverse(op,c[i],scale))
#                 vols.extend([v[i]]*4)
#                 harmo.extend([op-7, scale[high + max((abs(harmony[i]) // 2),4)],op, scale[low - max((abs(harmony[i]) // 2),4)]])
#                 harmoVols.extend([v[i]-15]*4)
#                 bassline.extend([bass, bass])
#                 melody.extend([scale[note],scale[note]-2,scale[note]-4,scale[note]-6])
#             else:
#                 arp.extend(traverse(op,c[i],scale))
#                 vols.extend([v[i]]*4)
#                 harmo.extend([op-7, scale[high + max((abs(harmony[i]) // 2),6)],op, scale[low - max((abs(harmony[i]) // 2),6)]])
#                 harmoVols.extend([v[i]-15]*4)
#                 bassline.extend([bass, bass])
#                 melody.extend([scale[note],scale[note]-2,scale[note]-4,scale[note]-6])
 
    arp2 = zip(arp,vols,harmo,harmoVols,durations,chrds)

    return (arp2, fit_to_scale(bassline, scale), zip(fit_to_scale(melody,scale),vols))




# In[32]:


def glide(start,end,scale,up=True):
    glide = [start]
    pos = scale.index(start)
    if up:
        while pos < scale.index(end):
            pos += 1
            glide.append(scale[pos])
            pos += 1
            glide.append(scale[pos])
            pos -= 1
            glide.append(scale[pos])
    return glide

    


# In[ ]:





# In[33]:


'''

~different types of midi output for different candles~

1 - Arpeggio
2 - Ladder
3 - Chord
4- 1 or 2 single notes




Arpeggio: O L C H when up, or OHCL when down



'''

def chord(note, scale,up, seventh=False):
    base = scale.index(fit_to_scale([note], scale)[0])
    if up > 0:
        if seventh:
            return [scale[base], scale[base+2], scale[base+4], scale[base+6]]
        else:
            return [scale[base], scale[base+2], scale[base+4],scale[base+8]]
    else:
        if seventh:
            return [scale[base], scale[base-2], scale[base-4], scale[base-6]]
        else:
            return [scale[base], scale[base-2], scale[base-4],scale[base-8]]

def traverse(start,end,scale):
    '''Function takes a start and end position as notes and a scale and creates a 4 note ladder-like path
    through them on the given scale'''
    breadth = abs(start-end)    
    jump = int(breadth/4)
    if start < end:
        ladder = [start,start+jump,start+(jump*2),start + (jump*3)]
    elif start > end:
        ladder = [end,end-jump,end-(jump*2),end - (jump*3)]
    else:
        ladder = [start,start+2,start-5,start+3]
    return fit_to_scale(ladder, scale)


# In[ ]:





# In[34]:


# def __init__(self,
#           numTracks=1,
#           removeDuplicates=True,
#           deinterleave=True,
#           adjust_origin=False,
#           file_format=1,
#           ticks_per_quarternote=TICKSPERQUARTERNOTE,
#           eventtime_is_ticks=False):


#red candles arpeggiate down and vice versa


# def compose_chart(chart, track, channel, time=0, duration, tempo, volume):
    


# degrees  = [40,62, 64, 65, 67, 69, 71, 72] # MIDI note number
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1/4   # In beats
tempo    = 100# In BPM
volume   = 80 # 0-127, as per the MIDI standard


qf = yf.download('TSLA',period='6mo', interval='1d')


MyMIDI = MIDIFile(2) # One track, defaults to format 1 (tempo track
                     # automatically created)
MIDIBass = MIDIFile(1)
MyMIDI.addTempo(track,time, tempo)
MIDIBass.addTempo(track,time, tempo)
Melody = MIDIFile(2)
Melody.addTempo(track,time,tempo)

for arp,vol,harmony,harmony_vol,dur,chrds in compose(treat_chart(qf, C_maj),C_maj)[0]:
    offset = humanize()
#     if chrds:
#         if time > 0:
#             time -= offset
#             MyMIDI.addNote(track, channel, int(arp), time, dur, int(vol))
#             MyMIDI.addNote(track, channel, int(harmony), time, dur, int(harmony_vol))
#             time= time + offset
#         else:
#             MyMIDI.addNote(track, channel, int(arp), time, dur, int(vol))
#             MyMIDI.addNote(track, channel, int(harmony), time, dur, 0)
#             time = time
#     else:
    if type(arp) == list:
        if time > 0:
            for note in arp:
                time -= offset
                MyMIDI.addNote(track, channel, int(note), time, dur, int(vol))
#                 MyMIDI.addNote(track, channel, int(harmony), time, dur, int(harmony_vol))
                time= time + offset
            time += dur
        else:
            for note in arp:
                MyMIDI.addNote(track, channel, int(note), time, dur, int(vol))
#                 MyMIDI.addNote(track, channel, int(harmony), time, dur, int(harmony_vol))
            time += dur 
    else:
        if time > 0:
            time -= offset
            MyMIDI.addNote(track, channel, int(arp), time, dur, int(vol))
            MyMIDI.addNote(track, channel, int(harmony), time, dur, int(harmony_vol))
            time= time + offset + dur
        else:
            MyMIDI.addNote(track, channel, int(arp), time, dur, int(vol))
            MyMIDI.addNote(track, channel, int(harmony), time, dur, int(harmony_vol))
            time += dur
    
            
time = 0
for note in compose(treat_chart(qf, C_maj),C_maj)[1]:
    MIDIBass.addNote(0, 0, note, time, 1/2, 70)
    time += 1
time = 0
for note,vol in compose(treat_chart(qf, C_maj),C_maj)[2]:
    offset = humanize()
    if time > 0:
        Melody.addNote(0, 0, int(note), time, duration, int(vol))
        time= time + (1/4)
    else:
        time -= offset
        Melody.addNote(0, 0, int(note), time, duration, int(vol))
        time= time + (1/4) + offset
    
with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
with open("bassline.mid", "wb") as output_file:
    MIDIBass.writeFile(output_file)
with open("melody.mid", "wb") as output_file:
    Melody.writeFile(output_file)


# In[81]:



# In[16]:


# #function to treat the ohlc
# #before rounding check for red/green candles and % distance of wicks
# #normalize volume to between 80-100


# #function to treat extreme volume
# def norm_vol(vol):
#     for i in vol:
#         if i == 0:
#             if statistics.median(vol) == 0:
#                 i = statistics.mean(vol)*1.1
#             else:
#                 i = 70 + choice(range(-4,4))
#     normalize(vol,30,60)

# # def zscore(df):
# #     z_scores = stats.zscore(df)
# #     abs_z_scores = np.abs(z_scores)
# #     filtered_entries = (abs_z_scores < 3).all(axis=1)
# #     removed = len(df) - len(filtered_entries)
# #     for i in df:
# #         if i not in filtered_entries:
# #             i = filtered_entries.max()
# #     #new_df = df[filtered_entries]
# #     return df    

# def his_los(df):
#     for i in range(len(df)):
#         df.Low[i] = min((df.Open[i],df.Close[i])) - df.Low[i]
#         df.High[i] = df.High[i] - max((df.Open[i],df.Close[i]))
#     normalize(df.High,4,1)
#     normalize(df.Low,4,1)
#     for i in range(len(df)):
#         df.Low[i] = int(df.Low[i])
#         df.High[i] = int(df.High[i])
#     for i in ('High','Low'):
#         df[i] = df[i].astype(int)
    
# def norm_dir(col):
#     for i in col:
#         i = abs(i)
#     return normalize(col,10)

 
# def treat_chart(chart, scale, chunks = 4):
    
# #     part = int(len(chart)/chunks)
# #     a = 0
# #     b = 1
#     if 'Adj Close' in chart.columns:
#             chart.Close = chart['Adj Close']
#             chart.drop(columns='Adj Close',inplace=True)
# #     chart['harmony'] = ""
    
    
#     norm_vol(chart.Volume)
#     high_low = chart['High'] - chart['Low']
#     high_close = np.abs(chart['High'] - chart['Close'].shift())
#     low_close = np.abs(chart['Low'] - chart['Close'].shift())
#     ranges = pd.concat([high_low, high_close, low_close], axis=1)
#     true_range = np.max(ranges, axis=1)
#     chart['atr'] = true_range.rolling(5).sum()/5
#     chart.drop(chart.head(5).index,inplace=True)
#     chart.drop(chart.tail(1).index,inplace=True)
#     chart.drop(chart.tail(len(chart)%chunks).index,inplace=True)
#     chart['up'] = chart.Open < chart.Close
#     his_los(chart)
#     chart['harmony'] = chart.Close - chart.Open
#     chart['harmony'] = normalize(chart['harmony'], 10, 1)
#     for column in ('Open', 'Close'):
#         chart[column] = normalize(chart[column],12, 60).round().astype(int)
#     chart.Open = fit_to_scale(chart.Open.astype(int), scale)
    
#     chart.harmony[chart.up == False] *= -1
#     chart.harmony = chart.harmony.astype(int)
#     chart.Close = fit_to_scale(chart.Open + chart.harmony, scale)
#     chart.High[chart.harmony > 0] = fit_to_scale(chart.Close + chart.High, scale)
#     chart.High[chart.harmony < 0] = fit_to_scale(chart.Open + chart.High, scale)
#     chart.Low[chart.harmony > 0] = fit_to_scale(chart.Open - chart.Low, scale)
#     chart.Low[chart.harmony < 0] = fit_to_scale(chart.Close - chart.Low, scale)
# #     chart['harmony'] = chart.Close - chart.Open
# #     for i in range(part):     
# #         chart.Open[a*chunks:b*chunks] = chart.Open[a*chunks:b*chunks].median()
        
# #         a += 1
# #         b += 1
#     chart['DC top'] = (chart.Close.rolling(window=20).max())
#     chart['DC bottom'] = (chart.Close.rolling(window=20).min())
#     chart['DC top'][:20] = chart.Close[:20].max()
#     chart['DC bottom'][:20] = chart.Close[:20].min()
#     for i in ['DC top', 'DC bottom']:
#         chart[i] = chart[i].astype(int)
#     chart['melody'] = round((chart['DC top'] + chart['DC bottom']) / 2)
#     chart['atr'] = normalize(chart.atr, 10).astype(int)
#     chart['new low'] = chart['DC bottom'].diff() < 0
#     chart['new high'] = chart['DC top'].diff() > 0
#     chart['new low'][0] = 0
#     chart['new high'][0] = 0
#     chart.melody[chart['harmony']<0] = ((chart['DC top'] + chart.melody)/2).astype(int)
#     chart.melody[chart['harmony']>0] = ((chart['DC bottom'] + chart.melody)/2).astype(int)
#     chart.melody[chart['new low']] = chart['DC bottom']
#     chart.melody[chart['new high']] = chart['DC top']
#     chart.melody = fit_to_scale(chart.melody.astype(int), scale)
#     chart['dur'] = [choice([1/4,1/2,1,1/4,1/2,1/3]) for i in range(len(chart))]
#     chart['pause'] = True
#     chart['pause'][::8] = False
#     chart['pause'][chart.atr > 2] = False
#     chart.Volume[chart.pause] = 1
#     return chart


# In[ ]:





# In[ ]:




