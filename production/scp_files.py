	#!/usr/bin/env python
	# encoding: utf-8
	from __future__ import with_statement
	import sys
	import subprocess
	import os

	def scp(source, server, path = ""):
		return not subprocess.Popen(["scp -r", source, "%s:%s" % (server, path)]).wait()

	def check_dest_ready(dest_path): 
	import os, errno

	try:
    	os.makedirs(dest_path)
    	print("CREATED dest_path",dest_path )
	except OSError as e:
		#pdb.set_trace()
    	if e.errno != errno.EEXIST:
        	raise
        if e.errno == errno.EEXIST:
        	print("dest_path already exists:",dest_path )


	def main(*args):
		filename_list = ["Subject_0003_17JUN1936", "Subject_0001_25NOV1940"]
		dest_server = "jaime@192.168.2.87"
		dest_path = "/Users/jaime/Downloads/"
		for ix, val in enumerate(filename_list):
			dest_path = os.path.join(dest_path, val)
			if check_dest_ready(dest_path):
				if scp(dest_path, server):
					print("File scp transferred successfully.")
					return 0
				else:
					print("File upload failed.")
				return 1
	if __name__ == "__main__": sys.exit(main(*sys.argv[1:]))
