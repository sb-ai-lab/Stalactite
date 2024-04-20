.. _batching_tutorial:

*how-to:* Make batchers
======================================
We need to remind that the Stalactite framework utilizes train/infer loops inside each agent: PartyMembers and PartyMaster.
And each of these loops contains its batchers, which have to be synchronized.

To make batch initialization you need to override the ``make_batcher`` method in your PartyMaster class.
For example, you can initialize batcher like this in ``stalactite/ml/honest/linear_regression/party_master.py`` or make your implementation.

.. code-block:: python

    def make_batcher(
                self,
                uids: Optional[List[str]] = None,
                party_members: Optional[List[str]] = None,
                is_infer: bool = False,
        ) -> Batcher:

        logger.info("Master %s: making a make_batcher for uids %s" % (self.id, len(uids)))
        self.check_if_ready()
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        epochs = 1 if is_infer else self.epochs
        if uids is None:
            raise RuntimeError('Master must initialize batcher with collected uids.')
        assert party_members is not None, "Master is trying to initialize make_batcher without members list"
        return ListBatcher(epochs=epochs, members=party_members, uids=uids, batch_size=batch_size)


Party members use their batchers, which need to be initialized in the PartyMember class.
You can find an example of such initialization in ``stalactite/ml/honest/base.py``.
The point is that your batchers have to have the same ``uids, batch_size, and epochs`` to make training/infer properly.

.. code-block:: python

    def make_batcher(
            self,
            uids: Optional[List[str]] = None,
            party_members: Optional[List[str]] = None,
            is_infer: bool = False,
    ) -> Batcher:
        epochs = 1 if is_infer else self.epochs
        batch_size = self._eval_batch_size if is_infer else self._batch_size
        uids_to_use = self._uids_to_use_test if is_infer else self._uids_to_use
        if uids_to_use is None:
            raise RuntimeError("Cannot create make_batcher, you must `register_records_uids` first.")
        return self._create_batcher(epochs=epochs, uids=uids_to_use, batch_size=batch_size)

    def _create_batcher(self, epochs: int, uids: List[str], batch_size: int) -> Batcher:
        """Create a make_batcher for training.

        :param epochs: Number of training epochs.
        :param uids: List of unique identifiers for dataset rows.
        :param batch_size: Size of the training batch.
        """
        logger.info("Member %s: making a make_batcher for uids" % self.id)
        if not self.is_consequently:
            return ListBatcher(epochs=epochs, members=None, uids=uids, batch_size=batch_size)
        else:
            return ConsecutiveListBatcher(
                epochs=epochs, members=self.members, uids=uids, batch_size=batch_size
            )
