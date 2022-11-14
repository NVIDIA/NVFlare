class Policy:
    pass


class Evidence:
    pass


class AttestationToken:
    pass


class PolicyRegistrationRequest:
    pass


class PolicyRegistrationResult:
    pass


class TokenRegistrationRequest:

    def __init__(self, identity: str, token: AttestationToken):
        self.identity = identity
        self.token = token


class TokenRegistrationResult:
    pass


class ClientRegistrationRequest:
    pass


class ClientRegistrationResult:
    pass


class TokenRetrievalRequest:
    pass


class TokenRetrievalResult:
    pass


class AttestationServiceAgent:

    """
    An AttestationServiceAgent is responsible for interacting with the Attestation Service (Verifier)
    """

    def __init__(self, service_endpoint):
        """

        Args:
            service_endpoint: the endpoint of the Attestation Service
        """
        pass

    def register_policy(
            self,
            identity: str,
            policy: Policy) -> PolicyRegistrationResult:
        """Register policy with the Attestation Service for the specified identity

        Args:
            identity: the unique identity that owns the policy
            policy: policy to be registered

        Returns: a PolicyRegistrationResult

        """
        pass

    def validate_policy(
            self,
            identity: str,
            evidence: Evidence) -> AttestationToken:
        """
        Request the Attestation Service to validate previously registered policy for the specified identity
        Args:
            identity: the unique identity that owns the policy
            evidence: evidence to support the policy claims

        Returns: an AttestationToken

        """
        pass

    def validate_token(
            self,
            identity: str,
            token: AttestationToken) -> Policy:
        """
        Request the Attestation Service to validate the specified token.

        Args:
            identity: the unique identity that the token is assigned to
            token: the token to be validated

        Returns: if valid, the claim policy of the identity

        """
        pass

    def validate_tokens(
            self,
            identity_tokens: dict) -> dict:
        """
        Request the Attestation Service to validate multiple tokens in one request.

        Args:
            identity_tokens: a dict of: identity => token

        Returns: a dict of identity => policy

        """
        pass


class AttestationOrchestrator:
    """
    AttestationOrchestrator implements orchestration service.
    Maybe embedded into FLARE Server, or into a separate server.
    The underlying server must provide communication capability

    Note that the AttestationOrchestrator requires an AttestationServiceAgent since it also
    needs to communicate with the Attestation Service
    """

    def __init__(self,
                 attestation_service_agent: AttestationServiceAgent):
        self.attestation_service_agent = attestation_service_agent

    def process_policy_registration(
            self,
            request: bytes) -> bytes:
        """Process policy_registration request from a client.
        This method is invoked by the underlying communication system.

        Args:
            request: the policy_registration request

        Returns: reply to this request

        When processing this request, it:
        - decodes the bytes to a PolicyRegistrationRequest object
        - validates the policy consistency
        - registers the policy with the Attestation Service and get a PolicyRegistrationResult
        - encodes the PolicyRegistrationResult to bytes
        - returns the encoded bytes

        Note: as far as the underlying server is concerned, the request and reply are just bytes.
        The bytes returned by this method will be delivered by the server to the requester (more precisely
        the AttestationOrchestratorAgent on the client).

        """
        pass

    def process_client_registration(
            self,
            request: bytes) -> bytes:
        """Process client_registration request from a client.
        This method is invoked by the underlying communication system.

        Args:
            request: the client_registration request

        Returns: reply to this request

        When processing this request, it:
        - decodes the bytes to a ClientRegistrationRequest object
        - validates the request
        - generates a ClientRegistrationResult (which contains a nonce)
        - encodes the ClientRegistrationResult to bytes
        - returns the encoded bytes

        Note: as far as the underlying server is concerned, the request and reply are just bytes.
        The bytes returned by this method will be delivered by the server to the requester (more precisely
        the AttestationOrchestratorAgent on the client).

        """
        pass

    def process_token_registration(
            self,
            request: bytes) -> bytes:
        """Process token_registration request from a client.
        This method is invoked by the underlying communication system.

        Args:
            request: the token_registration request

        Returns: reply to this request

        When processing this request, it:
        - decodes the bytes to a TokenRegistrationRequest
        - validates data
        - if valid, keep the token with the identity in a dict
        - generates a TokenRegistrationResult
        - encodes the TokenRegistrationResult to bytes
        - returns the encoded bytes

        Note: as far as the underlying server is concerned, the request and reply are just bytes.
        The bytes returned by this method will be delivered by the server to the requester (more precisely
        the AttestationOrchestratorAgent on the client).

        """
        pass

    def process_token_retrieval(
            self,
            request: bytes) -> bytes:
        """Process token_retrieval request from a client.
        This method is invoked by the underlying communication system.

        Args:
            request: the token_retrieval request

        Returns: reply to this request

        When processing this request, it:
        - decodes the bytes to a TokenRetrievalRequest object
        - validates the request
        - if valid, look up the token for the identity
        - generates a TokenRetrievalResult
        - encodes the TokenRetrievalResult to bytes
        - returns the encoded bytes

        Note: as far as the underlying server is concerned, the request and reply are just bytes.
        The bytes returned by this method will be delivered by the server to the requester (more precisely
        the AttestationOrchestratorAgent on the client).

        """
        pass

    def get_token(self, identity: str) -> AttestationToken:
        """ Get the token of the specified identity
        Find the token from the token-lookup table.

        Args:
            identity: the identity that the token belongs to

        Returns: token of the identity, or None if not found

        """
        pass


class Messenger:
    """
    This is the spec of a communicator that implements communication capability needed by
    the AttestationOrchestratorAgent.
    """

    def send(self, request: bytes, timeout: float=None) -> bytes:
        """Send the request to the AttestationOrchestrator

        Args:
            request:
            timeout: how long to wait for reply. If None, use default set by the messenger

        Returns: reply from the peer

        Exception will be raised when timed out or communication error

        """
        pass


class AttestationOrchestratorAgent:
    """
    AttestationOrchestratorAgent performs as an agent of the AttestationOrchestrator.
    The underlying server must provide communication capability

    Note that the AttestationOrchestratorAgent requires an AttestationServiceAgent since it
    needs to communicate with the Attestation Service
    """

    def __init__(
            self,
            my_identity: str,
            attestation_service_agent: AttestationServiceAgent,
            messenger: Messenger):
        """
        Constructor of the AttestationOrchestratorAgent

        Args:
            my_identity: identity of the client that the agent represents
            attestation_service_agent: the agent for talking to AttestationService
            messenger: the Messenger that is responsible for communication
        """
        self.my_identity = my_identity
        self.messenger = messenger
        self.attestation_service_agent = attestation_service_agent

    def register_client_policy(
            self,
            policy: Policy) -> PolicyRegistrationResult:
        """Register the client's policy with the AttestationOrchestrator.
        - create a PolicyRegistrationRequest object
        - serialize this object into bytes
        - call the messenger to send the registration request to the AttestationOrchestrator
        - decode received reply bytes into PolicyRegistrationResult
        - return the object

        Args:
            policy: the policy to be registered

        Returns: a PolicyRegistrationResult

        """
        pass

    def register_client(self) -> AttestationToken:
        """
        This method makes multiple interactions with AttestationOrchestrator and the Attestation Service
        - create a ClientRegistrationRequest object
        - serialize the object to bytes
        - call messenger.send() to send the request to AttestationOrchestrator
        - deserialize received bytes to ClientRegistrationResult object, which contains a nonce
        - call TMPQuote to generate evidence
        - create Evidence object from the nonce and evidence
        - call self.validate_evidence() to Attestation Service and receive an AttestationToken
        - return the AttestationToken object

        Returns: an AttestationToken object

        """
        pass

    def register_token(
            self,
            token: AttestationToken) -> TokenRegistrationResult:
        """
        Register the client's token with the AttestationOrchestrator.
        - create a TokenRegistrationRequest object
        - serialize it to bytes
        - call the messenger to send the request to the AttestationOrchestrator
        - deserialize the reply to TokenRegistrationResult object
        - return the TokenRegistrationResult object

        Args:
            token: the token to be registered

        Returns: a TokenRegistrationResult object

        """
        pass

    def retrieve_token(
            self,
            identity: str) -> AttestationToken:
        """
        Retrieve the attestation token of the specified identity.
        - create TokenRetrievalRequest object
        - serialize it to bytes
        - call the messenger to send the request to the AttestationOrchestrator
        - deserialize the reply to TokenRetrievalRequest object
        - return the token in the TokenRetrievalRequest object

        Args:
            identity: the identity of the token to be retrieved

        Returns: the token of the identity or None if not found

        """
        pass


"""
Utility Functions
"""


def load_policy(policy_definition_file: str) -> Policy:
    """
    Loads policy from the specified file (JSON, XML?) to a Policy object

    Args:
        policy_definition_file: the full path of the policy definition file

    Returns: a Policy object.

    Raise exception if file is invalid

    """
    pass


def check_claim(
        claim_policy: Policy,
        requirement_policy: Policy) -> bool:
    """
    Check whether the claim policy satisfies the specified requirements.

    Args:
        claim_policy: the policy to be checked
        requirement_policy: the policy that defines requirements

    Returns: True if claim policy meets requirements; False otherwise.

    """
    pass
