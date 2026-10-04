"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable


async def _async_noop_rollback():
    """项目内部接口说明。"""
    return []


class Checkpoints(LoggingConfigurable):
    """项目内部接口说明。"""

    #: Whether move_all_checkpoints tracks individual moves and supports
    #: rolling them back when the content rename fails afterwards.  Custom
    #: checkpoint backends inherit False and keep the legacy
    #: "content moves first" ordering until they opt in.
    supports_rename_rollback = False

    def create_checkpoint(self, contents_mgr, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def list_checkpoints(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def move_all_checkpoints(self, old_path, new_path):
        """Move all checkpoints of ``old_path`` to ``new_path``.

        Returns ``(moved_ids, rollback)``: ``moved_ids`` lists the
        checkpoints that were moved and ``rollback()`` reverses those moves,
        returning a list of errors that prevented the reversal.

        If a checkpoint subclass only implements the primitive
        :meth:`rename_checkpoint`, the moves are tracked and automatically
        rolled back when one of them fails (the content has not moved yet at
        that point).  A subclass that overrides
        :meth:`rename_all_checkpoints` keeps full control of the move; in
        that case no rollback information is available.
        """
        if type(self).rename_all_checkpoints is not Checkpoints.rename_all_checkpoints:
            # Backwards compatibility: a custom rename_all_checkpoints owns
            # the whole operation.
            self.rename_all_checkpoints(old_path, new_path)
            return [], list

        moved: list[str] = []
        try:
            for cp in self.list_checkpoints(old_path):
                checkpoint_id = cp["id"]
                self.rename_checkpoint(checkpoint_id, old_path, new_path)
                moved.append(checkpoint_id)
        except BaseException:
            # Undo the moves that already happened so the checkpoint store
            # is left in its original state wherever possible.
            errors = self._rollback_checkpoint_moves(moved, new_path, old_path)
            if errors:
                self.log.error(
                    "Checkpoint rename %s -> %s failed and rollback also "
                    "failed (%r); retrying the rename is safe",
                    old_path,
                    new_path,
                    errors,
                    exc_info=True,
                )
            raise

        def rollback() -> list[BaseException]:
            """Reverse completed moves; return errors that prevented it."""
            return self._rollback_checkpoint_moves(moved, new_path, old_path)

        return moved, rollback

    def _rollback_checkpoint_moves(self, moved_ids, src_path, dest_path) -> list[BaseException]:
        """Reverse a set of completed checkpoint moves, best effort."""
        errors: list[BaseException] = []
        for checkpoint_id in reversed(moved_ids):
            try:
                self.rename_checkpoint(checkpoint_id, src_path, dest_path)
            except Exception as e:
                errors.append(e)
        return errors

    def rename_all_checkpoints(self, old_path, new_path):
        """项目内部接口说明。"""
        for cp in self.list_checkpoints(old_path):
            self.rename_checkpoint(cp["id"], old_path, new_path)

    def delete_all_checkpoints(self, path):
        """项目内部接口说明。"""
        for checkpoint in self.list_checkpoints(path):
            self.delete_checkpoint(checkpoint["id"], path)


class GenericCheckpointsMixin:
    """项目内部接口说明。"""

    def create_checkpoint(self, contents_mgr, path):
        model = contents_mgr.get(path, content=True)
        type_ = model["type"]
        if type_ == "notebook":
            return self.create_notebook_checkpoint(
                model["content"],
                path,
            )
        elif type_ == "file":
            return self.create_file_checkpoint(
                model["content"],
                model["format"],
                path,
            )
        else:
            raise HTTPError(500, "Unexpected type %s" % type)

    def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        type_ = contents_mgr.get(path, content=False)["type"]
        if type_ == "notebook":
            model = self.get_notebook_checkpoint(checkpoint_id, path)
        elif type_ == "file":
            model = self.get_file_checkpoint(checkpoint_id, path)
        else:
            raise HTTPError(500, "Unexpected type %s" % type_)
        contents_mgr.save(model, path)

    # Required Methods
    def create_file_checkpoint(self, content, format, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def create_notebook_checkpoint(self, nb, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def get_file_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def get_notebook_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError


class AsyncCheckpoints(Checkpoints):
    """项目内部接口说明。"""

    async def create_checkpoint(self, contents_mgr, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def list_checkpoints(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def move_all_checkpoints(self, old_path, new_path):
        """Asynchronous counterpart of :meth:`Checkpoints.move_all_checkpoints`."""
        if type(self).rename_all_checkpoints is not AsyncCheckpoints.rename_all_checkpoints:
            await self.rename_all_checkpoints(old_path, new_path)
            return [], _async_noop_rollback

        moved: list[str] = []
        try:
            for cp in await self.list_checkpoints(old_path):
                checkpoint_id = cp["id"]
                await self.rename_checkpoint(checkpoint_id, old_path, new_path)
                moved.append(checkpoint_id)
        except BaseException:
            errors = await self._rollback_checkpoint_moves_async(moved, new_path, old_path)
            if errors:
                self.log.error(
                    "Checkpoint rename %s -> %s failed and rollback also "
                    "failed (%r); retrying the rename is safe",
                    old_path,
                    new_path,
                    errors,
                    exc_info=True,
                )
            raise

        async def rollback() -> list[BaseException]:
            """Reverse completed moves; return errors that prevented it."""
            return await self._rollback_checkpoint_moves_async(moved, new_path, old_path)

        return moved, rollback

    async def _rollback_checkpoint_moves_async(
        self, moved_ids, src_path, dest_path
    ) -> list[BaseException]:
        """Reverse a set of completed checkpoint moves, best effort."""
        errors: list[BaseException] = []
        for checkpoint_id in reversed(moved_ids):
            try:
                await self.rename_checkpoint(checkpoint_id, src_path, dest_path)
            except Exception as e:
                errors.append(e)
        return errors

    async def rename_all_checkpoints(self, old_path, new_path):
        """项目内部接口说明。"""
        for cp in await self.list_checkpoints(old_path):
            await self.rename_checkpoint(cp["id"], old_path, new_path)

    async def delete_all_checkpoints(self, path):
        """项目内部接口说明。"""
        for checkpoint in await self.list_checkpoints(path):
            await self.delete_checkpoint(checkpoint["id"], path)


class AsyncGenericCheckpointsMixin(GenericCheckpointsMixin):
    """项目内部接口说明。"""

    async def create_checkpoint(self, contents_mgr, path):
        model = await contents_mgr.get(path, content=True)
        type_ = model["type"]
        if type_ == "notebook":
            return await self.create_notebook_checkpoint(
                model["content"],
                path,
            )
        elif type_ == "file":
            return await self.create_file_checkpoint(
                model["content"],
                model["format"],
                path,
            )
        else:
            raise HTTPError(500, "Unexpected type %s" % type_)

    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """项目内部接口说明。"""
        content_model = await contents_mgr.get(path, content=False)
        type_ = content_model["type"]
        if type_ == "notebook":
            model = await self.get_notebook_checkpoint(checkpoint_id, path)
        elif type_ == "file":
            model = await self.get_file_checkpoint(checkpoint_id, path)
        else:
            raise HTTPError(500, "Unexpected type %s" % type_)
        await contents_mgr.save(model, path)

    # Required Methods
    async def create_file_checkpoint(self, content, format, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def create_notebook_checkpoint(self, nb, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def get_file_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def get_notebook_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        raise NotImplementedError
