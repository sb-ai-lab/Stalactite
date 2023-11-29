import logging
from queue import Queue
from party_master_Impl import PartyMasterImpl
from party_member_Impl import PartyMemberImpl
from party_Impl import PartyImpl

# logging.basicConfig(level=logging.DEBUG,
#                     format='(%(threadName)-9s) %(message)s',)

if __name__ == "__main__":

    # master = PartyMemberImpl()
    # members = [PartyMemberImpl() for _ in range(3)]
    master = 0
    members = [x for x in range(2)]

    party = PartyImpl(master, members)
    party.initialize()
    party.run()
    # master.run()




