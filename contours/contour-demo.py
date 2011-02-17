import cv, Image

## IMPORTANT! Change the following to point to the dir where you have
## the image "tubs-camview-perspective.png" and an empty space to play in.
IMAGEDIR = "."

# Use PIL to load an RGBA image and pass it to an OpenCV structure:
pi = Image.open('%s/tubs-camview-perspective.png' % IMAGEDIR)
p_im = cv.CreateImageHeader(pi.size, cv.IPL_DEPTH_8U, 4) # OpenCV Image header
cv.SetData(p_im, pi.tostring())

def sm(ipimage, name):
  # Save greyscale image
  # Get a file to write to:
  fp = open("%s/%s.png" % (IMAGEDIR, name), "wb")
  # The L mode here tells the PIL function how wide the OpenCV image data
  # is. 8 bit single channel, often a 'greyscale' of some sort, is "L"
  gi = Image.fromstring("L", cv.GetSize(ipimage), ipimage.tostring())
  # Save as a PNG
  gi.save(fp, format="PNG")
  # clean up
  fp.close()
  del gi

def srgb(im, name):
  # Save an RGB image - yeah should autodetect... exercise for the reader.
  fp = open("%s/%s.png" % (IMAGEDIR, name), "wb")
  gi = Image.fromstring("RGB", cv.GetSize(im), im.tostring())
  gi.save(fp, format="PNG")
  fp.close()
  del gi

def huethresh(srcrgb_im, hlower, hupper, slower, supper):
  # allocate some memory to do the conversion into. Otherwise, we'd
  # overwrite the source data and have to reload it. Colourspace conversions
  # aren't lossless typically!
  hsv = cv.CreateImage(cv.GetSize(srcrgb_im), cv.IPL_DEPTH_8U, 3)
  cv.Zero(hsv)
  # convert rgb to hsv
  cv.CvtColor(srcrgb_im, hsv, cv.CV_RGB2HSV)
  # Allocate some memory to hold each individual HSV channel
  h = cv.CreateImage(cv.GetSize(srcrgb_im), cv.IPL_DEPTH_8U, 1)
  s = cv.CreateImage(cv.GetSize(srcrgb_im), cv.IPL_DEPTH_8U, 1)
  v = cv.CreateImage(cv.GetSize(srcrgb_im), cv.IPL_DEPTH_8U, 1)
  # split rgb image into hsv planes:
  # Handy function, but requires you to pass in None for channels that aren't applicable
  # eg if you want to split a two channel image, you'd have to have ... , None, None) on the 
  # end
  cv.CvtPixToPlane(srcrgb_im, h, s, v, None)
  # threshold the planes within upper + lower limits
  cv.InRangeS(h, hlower, hupper, h)  # output values will be 0 or 255 (8bit)
  cv.InRangeS(s, slower, supper, s)
  # Logical AND the planes together, and store the result in v
  cv.And(h, s, v) # again, 0 or 255

  # As InRangeS + And are all done by the OpenCV C code, this is much faster
  # than iterating through each pixel in python
  del h, s, hsv
  return v

def rubbish_huesatthreshold(srcrgb_im, hlower, hupper, slower, supper):
  # Rubbish as the thresholding will be done in python instead
  hsv = cv.CreateImage(cv.GetSize(srcrgb_im), cv.IPL_DEPTH_8U, 3)
  # convert rgb to hsv
  cv.CvtColor(srcrgb_im, hsv, cv.CV_RGB2HSV)
  # Allocate some memory to hold each individual HSV channel
  h = cv.CreateImage(cv.GetSize(srcrgb_im), cv.IPL_DEPTH_8U, 1)
  s = cv.CreateImage(cv.GetSize(srcrgb_im), cv.IPL_DEPTH_8U, 1)
  v = cv.CreateImage(cv.GetSize(srcrgb_im), cv.IPL_DEPTH_8U, 1)
  # split rgb image into hsv planes:
  cv.CvtPixToPlane(srcrgb_im, h, s, v, None)
  # Thresholding...
  # expecting a 2D image
  (cols,rows) = cv.GetSize(srcrgb_im)
  # GOTCHA WARNING: per-pixel access is actually by (row, column), the opposite order from
  # the result of GetSize!
  # SECOND GOTCHA: pixels are accessed by passing in a tuple! Not h[x][y], but h[(x,y)]!
  # reusing v again so clearing it to be safe
  cv.Zero(v)
  for y in cols:
    for x in rows:
      h_val = h[(x,y)]
      s_val = s[(x,y)]
      if (h_val>=hlower and h_val<=hupper) and (s_val>=slower and s_val<=supper):
        v[(x,y)] = 255
      else:
        v[(x,y)] = 0
  del h, s, hsv
  return v

# Hue + Saturation thresholding original image.
# interested in green lids and the things embedded on them
t = huethresh(p_im, 0, 60, 140 ,255)

# Store a copy of this thresholded image for later examination
sm(t, "initial-thresholded.png")

seq = cv.FindContours(t, cv.CreateMemStorage(), cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE)

# Why CV_RETR_LIST? because it makes later navigation a little more straightforward.
# just walk along the seq.h_next() -> next seq until you hit the NULL point (None) to
# go through the list

# Draw a copy of the rough initial contours on a plain file and then on a clone of the
# thresholded pic. Going to use the 3 channels as HSV, and then convert to RGB as this
# makes it easier to change colours per contour:
plain_im = cv.CreateImage(pi.size, cv.IPL_DEPTH_8U, 3)  # 3 channel
# good habit to zero this memory
cv.Zero(plain_im)
t_clone_im = cv.CloneImage(t)

#iterate through the seq and draw it on plain and onto the clone:
s = seq
hue = 10
while(s):
  # (image to draw on, contour sequence, 'outer' contour colour, 'inner' colour, 
  #      follow contour seqeunces on (-1 = just draw this contour), line thickness)
  cv.DrawContours(plain_im, s, (hue,200,100), (hue,200,100), -1, 2)  # 3chan -> tuple
  cv.DrawContours(t_clone_im, s, 255, 255, -1, 2)  # 1 chan -> scalar
  hue = (hue+22) % 255
  s = s.h_next()

# convert HSV to RGB
cv.CvtColor(plain_im, plain_im, cv.CV_HSV2RGB)
#store
srgb(plain_im, "rough_contours_coloured")
sm(t_clone_im, "rough_contours_overlaid_on_thresh")

# clean up
del t_clone_im
# Going to reuse plain_im later...

# Make some approximate contours based on the 'found' contours
# You'll see the difference next.
approxseq = cv.ApproxPoly(seq, cv.CreateMemStorage(), cv.CV_POLY_APPROX_DP, 5, 1)

s = approxseq
approxlrg_seq = []
while(s.h_next()):
  if len(s) > 4: # make sure it has a significant number of points of interest
    # Might be worthwhile basing this on the area of the contour
    approxlrg_seq.append(s)
  s = s.h_next()

# Draw these approx contours out in thick blue, the originals in thin red
cv.Zero(plain_im)  # clear up previously used plain_im
# Draw 100 original contours at least, treating channels as RGB
cv.DrawContours(plain_im, seq, (200,0,0), (200,0,0), 100, 1)
# Draw all of the approx contours over the top
cv.DrawContours(plain_im, approxseq, (0,0,255), (0,0,255), 100, 2)
srgb(plain_im, "contour_comparison")

# now clean up
del plain_im

# If you want to see each contour in a greyscale image:
# 
#def explode_contours(seq_list, prefix, f):
#  for ind in xrange(len(seq_list)):
#    cv.Zero(f)
#    cv.DrawContours(f, seq_list[ind], 255, 255, -1, 2)
#    sm(f, "contourtest-%s-%s" % (prefix, ind))
#
#f = cv.CreateImage(cv.GetSize(t), cv.IPL_DEPTH_8U, 1)
#explode_contours(approxlrg_seq, "approxseq", f)
#del f

# pick the first contour as a src to match against:
src_shape = approxlrg_seq.pop(0)

i1, i2, i3 = [], [], []

# see the matches per contour, compared to the first one in the list
for ind in xrange(len(approxlrg_seq)):
  i1.append(cv.MatchShapes(src_shape, approxlrg_seq[ind], cv.CV_CONTOURS_MATCH_I1))
  i2.append(cv.MatchShapes(src_shape, approxlrg_seq[ind], cv.CV_CONTOURS_MATCH_I2))
  i3.append(cv.MatchShapes(src_shape, approxlrg_seq[ind], cv.CV_CONTOURS_MATCH_I3))
  print "src CV_CONTOURS_MATCH_I1 cmp to contour %s -> %s" % (ind, i1[ind])
  print "src CV_CONTOURS_MATCH_I2 cmp to contour %s -> %s" % (ind, i2[ind])
  print "src CV_CONTOURS_MATCH_I3 cmp to contour %s -> %s" % (ind, i3[ind])

# Normalise and combine matches to aid comparisions
def normalise_list(lst, to=1.0):
  m = max(lst)
  return [i/float(m)*to for i in lst]

i1 = normalise_list(i1)
i2 = normalise_list(i2)
i3 = normalise_list(i3)

for ind in xrange(len(approxlrg_seq)):
  print "src to contour %s -> %s" % (ind, i1[ind]+i2[ind]+i3[ind])

col_match = cv.CreateImage(cv.GetSize(t), cv.IPL_DEPTH_8U, 3)
cv.Zero(col_match)
for ind in xrange(len(approxlrg_seq)):
  if i1[ind]+i2[ind]+i3[ind] < 0.4:  # low difference in Hu Moments
    # FUDGE FACTOR ALERT!!!
    # 0.4 is arbitrary - picked as it is the level between the matches and the 
    # next highest false match (at 0.49, the circle around the bottom left diamond.)
    # Note that comparing Hu Moments like this is pretty loose and won't give great results, but
    # will likely separate out the matches.
    cv.DrawContours(col_match, approxlrg_seq[ind], (0,0,255), (0,0,255), -1, 3)  # Draw match in thick blue
  else:
    cv.DrawContours(col_match, approxlrg_seq[ind], (200,0,0), (200,0,0), -1, 1)  # Draw failed match in thin red

cv.DrawContours(col_match, src_shape, (0,255,0), (0,255,0), -1, 2)  # draw original in green
srgb(col_match, "matches")  # save image

