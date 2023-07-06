from pwl_writer import PWL

Vw = 4
t_write = 0.5e-6
t_step = t_write/1000

Vd = PWL(t_step=t_step, name="Vd")
Vg = PWL(t_step=t_step, name="Vg")
Vs = PWL(t_step=t_step, name="Vs")

###########################################################

def write(mem):
    if mem == 0:
        Vd.rect_pulse(duration=t_write, value=0)
        Vg.rect_pulse(duration=t_write, value=-Vw)
        Vs.rect_pulse(duration=t_write, value=0)
    elif mem == 1:
        Vd.rect_pulse(duration=t_write, value=0)
        Vg.rect_pulse(duration=t_write, value=Vw)
        Vs.rect_pulse(duration=t_write, value=0)
        
###########################################################

def disturb_row(mem):
    if mem == 0:
        Vd.rect_pulse(duration=t_write, value=-2*Vw/3)
        Vg.rect_pulse(duration=t_write, value=-Vw)
        Vs.rect_pulse(duration=t_write, value=-2*Vw/2)
    elif mem == 1:
        Vd.rect_pulse(duration=t_write, value=2*Vw/3)
        Vg.rect_pulse(duration=t_write, value=Vw)
        Vs.rect_pulse(duration=t_write, value=2*Vw/3)
        
###########################################################

def disturb_col(mem):
    if mem == 0:
        Vd.rect_pulse(duration=t_write, value=0)
        Vg.rect_pulse(duration=t_write, value=-Vw/3)
        Vs.rect_pulse(duration=t_write, value=0)
    elif mem == 1:
        Vd.rect_pulse(duration=t_write, value=0)
        Vg.rect_pulse(duration=t_write, value=Vw/3)
        Vs.rect_pulse(duration=t_write, value=0)
        
###########################################################

def disturb_off(mem):
    if mem == 0:
        Vd.rect_pulse(duration=t_write, value=-2*Vw/3)
        Vg.rect_pulse(duration=t_write, value=-Vw/3)
        Vs.rect_pulse(duration=t_write, value=-2*Vw/3)
    elif mem == 1:
        Vd.rect_pulse(duration=t_write, value=2*Vw/3)
        Vg.rect_pulse(duration=t_write, value=Vw/3)
        Vs.rect_pulse(duration=t_write, value=2*Vw/3)

###########################################################

def hold():
    Vd.rect_pulse(duration=t_write, value=0)
    Vg.rect_pulse(duration=t_write, value=0)
    Vs.rect_pulse(duration=t_write, value=0)

###########################################################
###########################################################
###########################################################

def pos_disturb_row(n=10):
    print("pos_disturb_row")
    
    hold()
    write(0)
    hold()
    
    for _ in range(n):
        disturb_row(1)
        hold()

###########################################################

def pos_disturb_col(n=10):
    print("pos_disturb_col")
    
    hold()
    write(0)
    hold()
    
    for _ in range(n):
        disturb_col(1)
        hold()

###########################################################

def pos_disturb_off(n=10):
    print("pos_disturb_off")
    
    hold()
    write(0)
    hold()
    
    for _ in range(n):
        disturb_off(0)
        hold()

###########################################################

def neg_disturb_row(n=10):
    print("neg_disturb_row")
    
    hold()
    write(1)
    hold()
    
    for _ in range(n):
        disturb_row(0)
        hold()

###########################################################

def neg_disturb_col(n=10):
    print("neg_disturb_col")
    
    hold()
    write(1)
    hold()
    
    for _ in range(n):
        disturb_col(0)
        hold()

###########################################################

def neg_disturb_off(n=10):
    print("neg_disturb_off")
   
    hold()
    write(1)
    hold()
    
    for _ in range(n):
        disturb_off(1)
        hold()

###########################################################
###########################################################
###########################################################

print("\n\n")

# Testar todos os tipos de disturbs
neg_disturb_row(20)
neg_disturb_col(20)
neg_disturb_off(20)

pos_disturb_row(20)
pos_disturb_col(20)
pos_disturb_off(20)

PWL.plot()

print(Vd)
print(Vg)
print(Vs)