# coding: utf-8
import sys
sys.path.append('..')
from init import Infrastructure

if __name__=="__main__":
    infra = Infrastructure()
    infra.compress_for_git()