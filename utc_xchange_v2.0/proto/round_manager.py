# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: proto/round_manager.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import Dict, Optional

import betterproto
import grpclib

from . import utc_bot


@dataclass
class BroadcastMessageRequest(betterproto.Message):
    """Request to broadcast a message to all competitors"""

    # The credentials used to request the broadcast containing {user, pw}
    creds: utc_bot.Credentials = betterproto.message_field(1)
    # The message to broadcast
    message: str = betterproto.string_field(2)


@dataclass
class BroadcastMessageResponse(betterproto.Message):
    """Response to request to broadcast event"""

    # Whether there was success in broadcasting the event to competitors
    ok: bool = betterproto.bool_field(1)
    # The message, if there is an error
    message: str = betterproto.string_field(2)


@dataclass
class BSUpdateRequest(betterproto.Message):
    """
    Request to update parameters associated with Black-Scholes calculations
    """

    # The credentials used to request the broadcast
    creds: utc_bot.Credentials = betterproto.message_field(1)
    # The parameters associated with each asset
    parameters: Dict[str, "BSUpdateRequestBSParameters"] = betterproto.map_field(
        2, betterproto.TYPE_STRING, betterproto.TYPE_MESSAGE
    )


@dataclass
class BSUpdateRequestBSParameters(betterproto.Message):
    """The parameters for B-S for a specific asset"""

    t: float = betterproto.double_field(1)
    q: float = betterproto.double_field(2)
    r: float = betterproto.double_field(3)


@dataclass
class BSUpdateResponse(betterproto.Message):
    """Response to request to update Black-Scholes info"""

    # Whether the BS update was successful
    ok: bool = betterproto.bool_field(1)
    # Message if the update wasn't ok
    message: str = betterproto.string_field(2)


@dataclass
class EndRoundRequest(betterproto.Message):
    """Request to end the round"""

    # The credentials used to request the broadcast
    creds: utc_bot.Credentials = betterproto.message_field(1)
    # The prices to mark to at the end of the round. Empty dict if none
    round_end_prices: Dict[str, str] = betterproto.map_field(
        2, betterproto.TYPE_STRING, betterproto.TYPE_STRING
    )


@dataclass
class EndRoundResponse(betterproto.Message):
    """Response to request to end the round"""

    # Whether the request to end the round succeeded
    ok: bool = betterproto.bool_field(1)
    # Message if the request failed for some reason
    message: str = betterproto.string_field(2)


@dataclass
class NumConnectedBotsRequest(betterproto.Message):
    """Request for number of connected bots"""

    # The credentials used to request the broadcast
    creds: utc_bot.Credentials = betterproto.message_field(1)


@dataclass
class NumConnectedBotsResponse(betterproto.Message):
    """Response to request for number of connected bots"""

    # Whether the request to get the # of connected bots succeeded
    ok: bool = betterproto.bool_field(1)
    # The number of connected bots
    num: int = betterproto.int32_field(2)


@dataclass
class ReadyRequest(betterproto.Message):
    """
    Sent by round managers when they've done all of their setup (handle_start
    has finished) and are ready for the round to start
    """

    # The credentials used to suggest that things are ready
    creds: utc_bot.Credentials = betterproto.message_field(1)


@dataclass
class ReadyResponse(betterproto.Message):
    """Response to Ready request"""

    pass


class RoundManagerServiceStub(betterproto.ServiceStub):
    """The gRPC Service"""

    async def broadcast_competition_event(
        self, *, creds: Optional[utc_bot.Credentials] = None, message: str = ""
    ) -> BroadcastMessageResponse:
        """Register a competitor"""

        request = BroadcastMessageRequest()
        if creds is not None:
            request.creds = creds
        request.message = message

        return await self._unary_unary(
            "/round_manager.RoundManagerService/BroadcastCompetitionEvent",
            request,
            BroadcastMessageResponse,
        )

    async def end_round(
        self,
        *,
        creds: Optional[utc_bot.Credentials] = None,
        round_end_prices: Optional[Dict[str, str]] = None,
    ) -> EndRoundResponse:
        """End the round"""

        request = EndRoundRequest()
        if creds is not None:
            request.creds = creds
        request.round_end_prices = round_end_prices

        return await self._unary_unary(
            "/round_manager.RoundManagerService/EndRound",
            request,
            EndRoundResponse,
        )

    async def b_s_update(
        self,
        *,
        creds: Optional[utc_bot.Credentials] = None,
        parameters: Optional[Dict[str, "BSUpdateRequestBSParameters"]] = None,
    ) -> BSUpdateResponse:
        """Black-Scholes update"""

        request = BSUpdateRequest()
        if creds is not None:
            request.creds = creds
        request.parameters = parameters

        return await self._unary_unary(
            "/round_manager.RoundManagerService/BSUpdate",
            request,
            BSUpdateResponse,
        )

    async def num_connected_bots_update(
        self, *, creds: Optional[utc_bot.Credentials] = None
    ) -> NumConnectedBotsResponse:
        """Check number of connected bots"""

        request = NumConnectedBotsRequest()
        if creds is not None:
            request.creds = creds

        return await self._unary_unary(
            "/round_manager.RoundManagerService/NumConnectedBotsUpdate",
            request,
            NumConnectedBotsResponse,
        )

    async def ready(
        self, *, creds: Optional[utc_bot.Credentials] = None
    ) -> ReadyResponse:
        request = ReadyRequest()
        if creds is not None:
            request.creds = creds

        return await self._unary_unary(
            "/round_manager.RoundManagerService/Ready",
            request,
            ReadyResponse,
        )
