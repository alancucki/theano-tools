#!/bin/bash
#
# Converts to torch7's serialized format
#
# Taken from https://github.com/locklin/torch-things.
# Attached license:
# Copyright (c) <2014> <Scott Locklin>
# 
# This software is provided 'as-is', without any express or implied
# warranty. In no event will the authors be held liable for any damages
# arising from the use of this software.
# 
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

if [ $# -ne 2 ]
then
    echo "Usage: $0 in out"
    echo "        Where in is the input csv and out is the torch file"
    echo "        The input csv must be a headerless 2d csv with the right number of carriage retuens"
    echo " To read into torch, use the following:"
    echo " file=torch.DiskFile(out,'r')"
    echo " csvdat=file:readObject()"
    exit 1
fi

echo "counting lines"
lines=$(wc -l < $1)
echo "lines = `echo $lines`"
cols=$(head -1 $1 | sed 's/[^,]//g' -| wc -c )
echo "columns = `echo $cols`"
nvals=$(($lines*$cols))
echo "creating header"
echo -e '4\n1\n3\nV 1\n18\ntorch.DoubleTensor\n2' > $2
echo "$lines $cols" >> $2
echo "$cols 1" >>$2
echo -e '1\n4\n2\n3\nV 1\n19\ntorch.DoubleStorage' >> $2
echo $nvals >> $2
echo "dumping data; this may take a moment..."
`cat $1 | tr -s ',\n' ' ' >> $2`
