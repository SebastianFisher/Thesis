# To get skillbridge to work:
# run in terminal
# skillbridge path
# run in CIW
# load("RESULT_OF_THIS^")
# pyRunScript "test_script.py" ?python "python3"

from skillbridge import Workspace
import random
from pprint import pprint

ws = Workspace.open("emx-test-server")
#print("cell view:", ws.ge.get_edit_cell_view())

cv = ws.ge.get_edit_cell_view()

# delete all nets in layour
# ws.db.delete_all_net(cv)

# Maybe: create a group, try to add all rectangles to group, then delete group??
#group = ws.db.create_group(cv, "EM Structure", ["collection", "unordered"])
em_structure = []
# Takes args d_cellViewID, t_groupName, l_groupType (list of collection/set, ordered/unordered, uniqueName/nonUniqueName

# print(ws.db.create_rect)
ex_grid = [[round(random.randint(0, 1)) for i in range(16)] for j in range(16)]

# expand rectangle a bit in each direction
RECT_SIZE = 10
EXTRA_LENGTH = 0.75

# Create grid in LD layer
for i in range(len(ex_grid)):
  for j in range(len(ex_grid[i])):
    if ex_grid[i][j] == 1:
      # Create rectangle at this point in grid and add it to the group
      bottom_left = [(j*RECT_SIZE)-EXTRA_LENGTH, (i*RECT_SIZE)-EXTRA_LENGTH]
      top_right = [(j*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH, (i*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH]
      rect = ws.db.create_rect(cv, ["LD", "drawing"], [bottom_left, top_right])

      em_structure.append(rect)

# Create underlying rectangle in M1_2B layer
under_rect = ws.db.create_rect(cv, ["M1_2B", "drawing"], [(-50, -50), (200, 200)])

# Create input port on LD and M1_2B in the same spot
# Just use closest 1 to middle i n grid
j = 0
i = len(ex_grid[i]) // 2

while i > 0:
  if ex_grid[i][j] == 1:
    break
  i -= 1

bottom_left = [(j*RECT_SIZE)-EXTRA_LENGTH, (i*RECT_SIZE)-EXTRA_LENGTH]
top_right = [(j*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH, (i*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH]
in_pin_ld = ws.db.create_rect(cv, ["LD", "pin"], [bottom_left, top_right])
#in_pin_ld.terminal = "IN1"


in_pin_m1 = ws.db.create_rect(cv, ["M1_2B", "pin"], [bottom_left, top_right])

# Create output port on LD and M1_2B in the same spot
# Use first 1 in the middle of far right column
j = len(ex_grid[0])-1
i = len(ex_grid) // 2

while i > 0:
  if ex_grid[i][j] == 1:
    break
  i -= 1

bottom_left = [(j*RECT_SIZE)-EXTRA_LENGTH, (i*RECT_SIZE)-EXTRA_LENGTH]
top_right = [(j*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH, (i*RECT_SIZE)+RECT_SIZE+EXTRA_LENGTH]
out_pin_ld = ws.db.create_rect(cv, ["LD", "pin"], [bottom_left, top_right])
out_pin_m1 = ws.db.create_rect(cv, ["M1_2B", "pin"], [bottom_left, top_right])

# print group
#pprint(vars(group))
#ws.db.delete_group_by_name(cv, "EM Structure")

#for rect in em_structure:
#  ws.db.delete_object(rect)

# Redraw the layout window
# ws.hi.redraw()

# First do leHiSave to save the changes?

# Simulating in EMX
#show_EMX_gui(nil nil)
#hiiSetCurrentForm('EMXform3')
#EMXform3->Start->value=30000000000.0
#EMXform3->Stop->value=100000000000.0
#EMXform3->Step->value=1000000000.0
#EMXform3->Signals->value="*"

#EMXform3->Cadence_pins->value= t
#EMXform3->Label_depth->value=1







