from party_master_Impl import PartyMasterImpl
from party_member_Impl import PartyMemberImpl
from party_Impl import PartyImpl


if __name__ == "__main__":

    master = PartyMasterImpl(epochs=1)
    members = [PartyMemberImpl() for _ in range(2)]
    party = PartyImpl(master, members)
    party.initialize()
    party.run()
