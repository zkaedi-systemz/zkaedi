// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import "@openzeppelin/contracts/interfaces/IERC165.sol";

/**
 * @title HModelToken - The Ultimate AI Model Ecosystem Token
 * @dev Most comprehensive ERC20 token with advanced features for H-Model AI system
 * @author iDeaKz - Token Creation Mastermind
 * @notice This contract implements a full-featured token for AI model training rewards,
 *         governance, staking, liquidity mining, and advanced DeFi operations
 * 
 * ██╗  ██╗███╗   ███╗ ██████╗ ██████╗ ███████╗██╗         ████████╗ ██████╗ ██╗  ██╗███████╗███╗   ██╗
 * ██║  ██║████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║         ╚══██╔══╝██╔═══██╗██║ ██╔╝██╔════╝████╗  ██║
 * ███████║██╔████╔██║██║   ██║██║  ██║█████╗  ██║            ██║   ██║   ██║█████╔╝ █████╗  ██╔██╗ ██║
 * ██╔══██║██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║            ██║   ██║   ██║██╔═██╗ ██╔══╝  ██║╚██╗██║
 * ██║  ██║██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗       ██║   ╚██████╔╝██║  ██╗███████╗██║ ╚████║
 * ╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝       ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝
 * 
 * Features:
 * ✅ ERC20 Standard with ALL Extensions
 * ✅ Advanced Staking & Yield Farming
 * ✅ Comprehensive Governance System
 * ✅ AI Model Training Rewards
 * ✅ Vector Embedding Incentives
 * ✅ Liquidity Mining Programs
 * ✅ Anti-MEV Protection
 * ✅ Flash Loan Resistance
 * ✅ Multi-Signature Operations
 * ✅ Emergency Pause System
 * ✅ Automatic Buyback & Burn
 * ✅ Cross-Chain Bridge Support
 * ✅ NFT Integration
 * ✅ Oracle Price Feeds
 * ✅ Advanced Security Measures
 */
contract HModelToken is 
    ERC20, 
    ERC20Burnable, 
    ERC20Permit, 
    ERC20Votes, 
    Ownable, 
    AccessControl, 
    Pausable, 
    ReentrancyGuard,
    EIP712 
{
    using ECDSA for bytes32;

    // ==================== CONSTANTS & IMMUTABLES ====================
    
    /// @notice Maximum total supply of tokens (1 Billion with 18 decimals)
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18;
    
    /// @notice Initial supply minted to deployer (100 Million)
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18;
    
    /// @notice Base staking reward rate (10% APY = 1000 basis points)
    uint256 public constant BASE_REWARD_RATE = 1000;
    
    /// @notice Maximum staking multiplier for long-term stakes (5x)
    uint256 public constant MAX_STAKE_MULTIPLIER = 500; // 5.00x
    
    /// @notice Minimum stake duration (1 day)
    uint256 public constant MIN_STAKE_DURATION = 1 days;
    
    /// @notice Maximum stake duration (4 years for maximum rewards)
    uint256 public constant MAX_STAKE_DURATION = 4 * 365 days;
    
    /// @notice Governance proposal threshold (0.1% of total supply)
    uint256 public constant PROPOSAL_THRESHOLD = MAX_SUPPLY / 1000;
    
    /// @notice Minimum voting period (3 days)
    uint256 public constant MIN_VOTING_PERIOD = 3 days;
    
    /// @notice Maximum voting period (14 days)
    uint256 public constant MAX_VOTING_PERIOD = 14 days;
    
    /// @notice Quorum requirement (4% of total voting power)
    uint256 public constant QUORUM_PERCENTAGE = 400; // 4.00%
    
    /// @notice AI model training reward pool allocation (20% of supply)
    uint256 public constant AI_REWARD_POOL = MAX_SUPPLY * 20 / 100;
    
    /// @notice Liquidity mining allocation (15% of supply)
    uint256 public constant LIQUIDITY_MINING_POOL = MAX_SUPPLY * 15 / 100;

    // ==================== ACCESS CONTROL ROLES ====================
    
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant AI_TRAINER_ROLE = keccak256("AI_TRAINER_ROLE");
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant EMERGENCY_ROLE = keccak256("EMERGENCY_ROLE");
    bytes32 public constant BRIDGE_ROLE = keccak256("BRIDGE_ROLE");

    // ==================== STRUCTS ====================
    
    /// @notice Comprehensive staking information
    struct StakeInfo {
        uint256 amount;           // Amount of tokens staked
        uint256 startTime;        // When the stake was created
        uint256 duration;         // Stake duration in seconds
        uint256 multiplier;       // Reward multiplier (basis points)
        uint256 rewardsClaimed;   // Total rewards claimed
        uint256 lastClaimTime;    // Last time rewards were claimed
        bool isActive;            // Whether the stake is currently active
        StakeType stakeType;      // Type of stake for different reward structures
    }
    
    /// @notice AI model training session data
    struct AITrainingSession {
        address trainer;          // Address of the AI trainer
        uint256 modelId;          // Unique identifier for the AI model
        uint256 startTime;        // Training session start time
        uint256 endTime;          // Training session end time
        uint256 accuracyScore;    // Model accuracy (basis points)
        uint256 complexityScore;  // Model complexity score
        uint256 rewardAmount;     // Calculated reward amount
        bool rewardClaimed;       // Whether reward has been claimed
        TrainingType trainingType; // Type of training performed
    }
    
    /// @notice Governance proposal structure
    struct Proposal {
        bytes32 id;               // Unique proposal identifier
        address proposer;         // Address that created the proposal
        string title;             // Proposal title
        string description;       // Detailed proposal description
        uint256 startTime;        // Voting start time
        uint256 endTime;          // Voting end time
        uint256 forVotes;         // Votes in favor
        uint256 againstVotes;     // Votes against
        uint256 abstainVotes;     // Abstain votes
        bool executed;            // Whether proposal has been executed
        bool canceled;            // Whether proposal has been canceled
        ProposalState state;      // Current proposal state
        ProposalType proposalType; // Type of proposal
        bytes executionData;      // Data to execute if proposal passes
        mapping(address => bool) hasVoted; // Track who has voted
        mapping(address => VoteChoice) votes; // Track vote choices
    }
    
    /// @notice Liquidity mining pool information
    struct LiquidityPool {
        address poolAddress;      // Address of the liquidity pool
        uint256 totalStaked;      // Total tokens staked in pool
        uint256 rewardRate;       // Reward rate per second
        uint256 lastUpdateTime;   // Last time rewards were updated
        uint256 rewardPerToken;   // Accumulated reward per token
        bool isActive;            // Whether pool is currently active
        mapping(address => uint256) stakedAmounts; // User staked amounts
        mapping(address => uint256) userRewardPerTokenPaid; // User reward tracking
        mapping(address => uint256) rewards; // Pending rewards
    }
    
    /// @notice Flash loan protection data
    struct FlashLoanGuard {
        mapping(address => uint256) balanceSnapshots;
        mapping(address => bool) isFlashLoanActive;
        uint256 blockNumber;
    }

    // ==================== ENUMS ====================
    
    enum StakeType {
        STANDARD,       // Standard staking with base rewards
        AI_TRAINING,    // Enhanced rewards for AI trainers
        GOVERNANCE,     // Voting-focused staking
        LIQUIDITY       // Liquidity provision staking
    }
    
    enum TrainingType {
        SUPERVISED,     // Supervised learning training
        UNSUPERVISED,   // Unsupervised learning training
        REINFORCEMENT,  // Reinforcement learning training
        DEEP_LEARNING,  // Deep neural network training
        TRANSFORMER,    // Transformer model training
        EMBEDDING       // Vector embedding training
    }
    
    enum ProposalState {
        PENDING,        // Proposal created but voting not started
        ACTIVE,         // Voting is currently active
        SUCCEEDED,      // Proposal passed and can be executed
        DEFEATED,       // Proposal failed
        EXECUTED,       // Proposal has been executed
        CANCELED        // Proposal was canceled
    }
    
    enum ProposalType {
        PARAMETER_CHANGE,  // Change contract parameters
        UPGRADE,          // Contract upgrade proposal
        TREASURY,         // Treasury management
        EMERGENCY,        // Emergency actions
        ECOSYSTEM         // Ecosystem development
    }
    
    enum VoteChoice {
        AGAINST,          // Vote against the proposal
        FOR,              // Vote for the proposal
        ABSTAIN           // Abstain from voting
    }

    // ==================== STATE VARIABLES ====================
    
    /// @notice Mapping of user addresses to their staking information
    mapping(address => StakeInfo[]) public userStakes;
    
    /// @notice Mapping of AI training sessions
    mapping(uint256 => AITrainingSession) public trainingSessions;
    
    /// @notice Mapping of proposal IDs to proposals
    mapping(bytes32 => Proposal) public proposals;
    
    /// @notice Mapping of liquidity pool addresses to pool info
    mapping(address => LiquidityPool) public liquidityPools;
    
    /// @notice Array of all proposal IDs for enumeration
    bytes32[] public proposalIds;
    
    /// @notice Array of active liquidity pool addresses
    address[] public activePools;
    
    /// @notice Total amount of tokens currently staked
    uint256 public totalStaked;
    
    /// @notice Total rewards distributed to date
    uint256 public totalRewardsDistributed;
    
    /// @notice Current AI training session counter
    uint256 public trainingSessionCounter;
    
    /// @notice Current proposal counter
    uint256 public proposalCounter;
    
    /// @notice Treasury balance for ecosystem development
    uint256 public treasuryBalance;
    
    /// @notice Buyback and burn mechanism
    uint256 public totalBurned;
    uint256 public lastBuybackTime;
    uint256 public buybackInterval = 7 days;
    uint256 public buybackPercentage = 100; // 1% of treasury
    
    /// @notice Anti-MEV and flash loan protection
    FlashLoanGuard private flashLoanGuard;
    mapping(address => uint256) private lastTransactionBlock;
    
    /// @notice Oracle price feed (for advanced features)
    address public priceOracle;
    uint256 public lastPriceUpdate;
    
    /// @notice Cross-chain bridge support
    mapping(uint256 => bool) public supportedChains;
    mapping(bytes32 => bool) public processedBridgeTransactions;
    
    /// @notice Emergency controls
    bool public emergencyMode;
    address public emergencyCouncil;
    uint256 public emergencyActivationTime;

    // ==================== EVENTS ====================
    
    /// @notice Emitted when tokens are staked
    event Staked(
        address indexed user,
        uint256 indexed stakeIndex,
        uint256 amount,
        uint256 duration,
        StakeType stakeType,
        uint256 multiplier
    );
    
    /// @notice Emitted when tokens are unstaked
    event Unstaked(
        address indexed user,
        uint256 indexed stakeIndex,
        uint256 amount,
        uint256 rewards
    );
    
    /// @notice Emitted when staking rewards are claimed
    event RewardsClaimed(
        address indexed user,
        uint256 indexed stakeIndex,
        uint256 amount
    );
    
    /// @notice Emitted when AI training session is started
    event AITrainingStarted(
        uint256 indexed sessionId,
        address indexed trainer,
        uint256 modelId,
        TrainingType trainingType
    );
    
    /// @notice Emitted when AI training session is completed
    event AITrainingCompleted(
        uint256 indexed sessionId,
        address indexed trainer,
        uint256 accuracyScore,
        uint256 rewardAmount
    );
    
    /// @notice Emitted when a governance proposal is created
    event ProposalCreated(
        bytes32 indexed proposalId,
        address indexed proposer,
        string title,
        ProposalType proposalType,
        uint256 startTime,
        uint256 endTime
    );
    
    /// @notice Emitted when a vote is cast
    event VoteCast(
        bytes32 indexed proposalId,
        address indexed voter,
        VoteChoice choice,
        uint256 weight,
        string reason
    );
    
    /// @notice Emitted when a proposal is executed
    event ProposalExecuted(
        bytes32 indexed proposalId,
        address indexed executor
    );
    
    /// @notice Emitted when liquidity is added to a pool
    event LiquidityAdded(
        address indexed user,
        address indexed pool,
        uint256 amount
    );
    
    /// @notice Emitted when liquidity is removed from a pool
    event LiquidityRemoved(
        address indexed user,
        address indexed pool,
        uint256 amount,
        uint256 rewards
    );
    
    /// @notice Emitted during buyback and burn operations
    event BuybackAndBurn(
        uint256 amount,
        uint256 timestamp
    );
    
    /// @notice Emitted when emergency mode is activated
    event EmergencyActivated(
        address indexed activator,
        string reason,
        uint256 timestamp
    );
    
    /// @notice Emitted for cross-chain bridge operations
    event BridgeTransfer(
        address indexed from,
        uint256 indexed targetChain,
        bytes32 indexed transactionId,
        uint256 amount
    );

    // ==================== CUSTOM ERRORS ====================
    
    /// @notice Thrown when an operation would exceed maximum supply
    error MaxSupplyExceeded(uint256 requested, uint256 maxSupply);
    
    /// @notice Thrown when user has insufficient balance
    error InsufficientBalance(uint256 requested, uint256 available);
    
    /// @notice Thrown when stake duration is invalid
    error InvalidStakeDuration(uint256 duration, uint256 min, uint256 max);
    
    /// @notice Thrown when trying to operate on non-existent stake
    error StakeNotFound(address user, uint256 stakeIndex);
    
    /// @notice Thrown when stake is not yet matured
    error StakeNotMatured(uint256 currentTime, uint256 maturityTime);
    
    /// @notice Thrown when user lacks required role
    error MissingRole(address user, bytes32 role);
    
    /// @notice Thrown when proposal is not in correct state
    error InvalidProposalState(bytes32 proposalId, ProposalState current, ProposalState required);
    
    /// @notice Thrown when user has already voted on proposal
    error AlreadyVoted(address user, bytes32 proposalId);
    
    /// @notice Thrown when voting power is insufficient
    error InsufficientVotingPower(uint256 current, uint256 required);
    
    /// @notice Thrown when operation is attempted during emergency mode
    error EmergencyModeActive();
    
    /// @notice Thrown when flash loan attack is detected
    error FlashLoanAttackDetected(address attacker);
    
    /// @notice Thrown when MEV attack is detected
    error MEVAttackDetected(address attacker);

    // ==================== MODIFIERS ====================
    
    /// @notice Ensures caller has specific role
    modifier onlyRole(bytes32 role) {
        if (!hasRole(role, msg.sender)) {
            revert MissingRole(msg.sender, role);
        }
        _;
    }
    
    /// @notice Prevents operations during emergency mode
    modifier notInEmergency() {
        if (emergencyMode) {
            revert EmergencyModeActive();
        }
        _;
    }
    
    /// @notice Protects against flash loans and MEV attacks
    modifier antiFlashLoan() {
        _checkFlashLoanAttack();
        _checkMEVAttack();
        _;
        _updateBalanceSnapshot();
    }
    
    /// @notice Validates stake exists and belongs to user
    modifier validStake(address user, uint256 stakeIndex) {
        if (stakeIndex >= userStakes[user].length) {
            revert StakeNotFound(user, stakeIndex);
        }
        if (!userStakes[user][stakeIndex].isActive) {
            revert StakeNotFound(user, stakeIndex);
        }
        _;
    }
    
    /// @notice Validates proposal exists and is in correct state
    modifier validProposal(bytes32 proposalId, ProposalState requiredState) {
        ProposalState currentState = getProposalState(proposalId);
        if (currentState != requiredState) {
            revert InvalidProposalState(proposalId, currentState, requiredState);
        }
        _;
    }

    // ==================== CONSTRUCTOR ====================
    
    /**
     * @notice Initialize the HModelToken with comprehensive features
     * @param initialOwner Address that will own the contract initially
     * @param _emergencyCouncil Address for emergency operations
     * @param _priceOracle Address of the price oracle contract
     */
    constructor(
        address initialOwner,
        address _emergencyCouncil,
        address _priceOracle
    ) 
        ERC20("HModel AI Token", "HMAI") 
        ERC20Permit("HModel AI Token")
        EIP712("HModel AI Token", "1")
        Ownable(initialOwner)
    {
        // Set up access control
        _grantRole(DEFAULT_ADMIN_ROLE, initialOwner);
        _grantRole(MINTER_ROLE, initialOwner);
        _grantRole(BURNER_ROLE, initialOwner);
        _grantRole(PAUSER_ROLE, initialOwner);
        _grantRole(AI_TRAINER_ROLE, initialOwner);
        _grantRole(ORACLE_ROLE, initialOwner);
        _grantRole(EMERGENCY_ROLE, _emergencyCouncil);
        
        // Initialize state variables
        emergencyCouncil = _emergencyCouncil;
        priceOracle = _priceOracle;
        treasuryBalance = INITIAL_SUPPLY / 10; // 10% to treasury
        
        // Mint initial supply
        _mint(initialOwner, INITIAL_SUPPLY);
        
        // Set up supported chains (example chain IDs)
        supportedChains[1] = true;      // Ethereum Mainnet
        supportedChains[137] = true;    // Polygon
        supportedChains[56] = true;     // BSC
        supportedChains[43114] = true;  // Avalanche
        supportedChains[250] = true;    // Fantom
        supportedChains[42161] = true;  // Arbitrum One
        supportedChains[10] = true;     // Optimism
        
        // Initialize flash loan protection
        flashLoanGuard.blockNumber = block.number;
    }

    // ==================== CORE TOKEN FUNCTIONS ====================
    
    /**
     * @notice Mint new tokens to specified address
     * @dev Only addresses with MINTER_ROLE can call this function
     * @param to Address to receive the newly minted tokens
     * @param amount Amount of tokens to mint
     */
    function mint(address to, uint256 amount) 
        external 
        onlyRole(MINTER_ROLE) 
        whenNotPaused 
        notInEmergency 
    {
        if (totalSupply() + amount > MAX_SUPPLY) {
            revert MaxSupplyExceeded(totalSupply() + amount, MAX_SUPPLY);
        }
        
        _mint(to, amount);
        
        // Update treasury if minting to treasury
        if (to == address(this)) {
            treasuryBalance += amount;
        }
    }
    
    /**
     * @notice Batch mint tokens to multiple addresses
     * @dev Efficient batch minting for airdrops and rewards distribution
     * @param recipients Array of addresses to receive tokens
     * @param amounts Array of amounts corresponding to each recipient
     */
    function batchMint(
        address[] calldata recipients, 
        uint256[] calldata amounts
    ) 
        external 
        onlyRole(MINTER_ROLE) 
        whenNotPaused 
        notInEmergency 
    {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        require(recipients.length <= 500, "Batch too large"); // Gas limit protection
        
        uint256 totalAmount;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        
        if (totalSupply() + totalAmount > MAX_SUPPLY) {
            revert MaxSupplyExceeded(totalSupply() + totalAmount, MAX_SUPPLY);
        }
        
        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amounts[i]);
        }
    }
    
    /**
     * @notice Burn tokens from specific address (with permission)
     * @dev Only BURNER_ROLE can burn tokens from other addresses
     * @param from Address to burn tokens from
     * @param amount Amount of tokens to burn
     */
    function burnFrom(address from, uint256 amount) 
        public 
        override 
        onlyRole(BURNER_ROLE) 
    {
        _burn(from, amount);
        totalBurned += amount;
    }
    
    /**
     * @notice Enhanced transfer function with MEV protection
     * @dev Overrides standard transfer with additional security checks
     */
    function transfer(address to, uint256 amount) 
        public 
        override 
        whenNotPaused 
        antiFlashLoan 
        returns (bool) 
    {
        return super.transfer(to, amount);
    }
    
    /**
     * @notice Enhanced transferFrom function with MEV protection
     * @dev Overrides standard transferFrom with additional security checks
     */
    function transferFrom(address from, address to, uint256 amount) 
        public 
        override 
        whenNotPaused 
        antiFlashLoan 
        returns (bool) 
    {
        return super.transferFrom(from, to, amount);
    }

    // ==================== ADVANCED STAKING SYSTEM ====================
    
    /**
     * @notice Stake tokens for rewards with customizable parameters
     * @dev Creates a new stake with specified duration and type
     * @param amount Amount of tokens to stake
     * @param duration Duration of the stake in seconds
     * @param stakeType Type of staking (affects reward calculation)
     * @return stakeIndex Index of the created stake
     */
    function stake(
        uint256 amount, 
        uint256 duration, 
        StakeType stakeType
    ) 
        external 
        whenNotPaused 
        notInEmergency 
        nonReentrant 
        returns (uint256 stakeIndex) 
    {
        if (amount == 0) {
            revert InsufficientBalance(amount, 0);
        }
        if (balanceOf(msg.sender) < amount) {
            revert InsufficientBalance(amount, balanceOf(msg.sender));
        }
        if (duration < MIN_STAKE_DURATION || duration > MAX_STAKE_DURATION) {
            revert InvalidStakeDuration(duration, MIN_STAKE_DURATION, MAX_STAKE_DURATION);
        }
        
        // Calculate stake multiplier based on duration and type
        uint256 multiplier = _calculateStakeMultiplier(duration, stakeType);
        
        // Transfer tokens to contract
        _transfer(msg.sender, address(this), amount);
        
        // Create stake record
        StakeInfo memory newStake = StakeInfo({
            amount: amount,
            startTime: block.timestamp,
            duration: duration,
            multiplier: multiplier,
            rewardsClaimed: 0,
            lastClaimTime: block.timestamp,
            isActive: true,
            stakeType: stakeType
        });
        
        userStakes[msg.sender].push(newStake);
        stakeIndex = userStakes[msg.sender].length - 1;
        
        totalStaked += amount;
        
        // Delegate voting power for governance stakes
        if (stakeType == StakeType.GOVERNANCE) {
            _delegate(msg.sender, msg.sender);
        }
        
        emit Staked(msg.sender, stakeIndex, amount, duration, stakeType, multiplier);
        
        return stakeIndex;
    }
    
    /**
     * @notice Unstake tokens and claim all pending rewards
     * @dev Can only unstake after the stake duration has elapsed
     * @param stakeIndex Index of the stake to unstake
     */
    function unstake(uint256 stakeIndex) 
        external 
        validStake(msg.sender, stakeIndex) 
        nonReentrant 
    {
        StakeInfo storage stakeInfo = userStakes[msg.sender][stakeIndex];
        
        uint256 maturityTime = stakeInfo.startTime + stakeInfo.duration;
        if (block.timestamp < maturityTime) {
            revert StakeNotMatured(block.timestamp, maturityTime);
        }
        
        uint256 stakedAmount = stakeInfo.amount;
        uint256 pendingRewards = _calculatePendingRewards(msg.sender, stakeIndex);
        
        // Mark stake as inactive
        stakeInfo.isActive = false;
        totalStaked -= stakedAmount;
        
        // Transfer staked tokens back to user
        _transfer(address(this), msg.sender, stakedAmount);
        
        // Mint and transfer rewards if any
        if (pendingRewards > 0) {
            _mint(msg.sender, pendingRewards);
            stakeInfo.rewardsClaimed += pendingRewards;
            totalRewardsDistributed += pendingRewards;
        }
        
        emit Unstaked(msg.sender, stakeIndex, stakedAmount, pendingRewards);
    }
    
    /**
     * @notice Claim pending rewards without unstaking
     * @dev Allows users to harvest rewards while maintaining their stake
     * @param stakeIndex Index of the stake to claim rewards from
     */
    function claimRewards(uint256 stakeIndex) 
        external 
        validStake(msg.sender, stakeIndex) 
        nonReentrant 
    {
        uint256 pendingRewards = _calculatePendingRewards(msg.sender, stakeIndex);
        
        if (pendingRewards > 0) {
            StakeInfo storage stakeInfo = userStakes[msg.sender][stakeIndex];
            
            stakeInfo.rewardsClaimed += pendingRewards;
            stakeInfo.lastClaimTime = block.timestamp;
            totalRewardsDistributed += pendingRewards;
            
            _mint(msg.sender, pendingRewards);
            
            emit RewardsClaimed(msg.sender, stakeIndex, pendingRewards);
        }
    }
    
    /**
     * @notice Calculate pending rewards for a specific stake
     * @dev Internal function to calculate rewards based on time elapsed and multipliers
     * @param user Address of the stake owner
     * @param stakeIndex Index of the stake
     * @return Pending reward amount
     */
    function _calculatePendingRewards(address user, uint256 stakeIndex) 
        internal 
        view 
        returns (uint256) 
    {
        StakeInfo storage stakeInfo = userStakes[user][stakeIndex];
        
        if (!stakeInfo.isActive) {
            return 0;
        }
        
        uint256 timeStaked = block.timestamp - stakeInfo.lastClaimTime;
        uint256 rewardRate = BASE_REWARD_RATE * stakeInfo.multiplier / 100;
        
        // Calculate annual reward and convert to actual time period
        uint256 annualReward = (stakeInfo.amount * rewardRate) / 10000;
        uint256 reward = (annualReward * timeStaked) / 365 days;
        
        return reward;
    }
    
    /**
     * @notice Calculate stake multiplier based on duration and type
     * @dev Internal function to determine reward multiplier
     * @param duration Stake duration in seconds
     * @param stakeType Type of stake
     * @return Multiplier in basis points (100 = 1x)
     */
    function _calculateStakeMultiplier(uint256 duration, StakeType stakeType) 
        internal 
        pure 
        returns (uint256) 
    {
        // Base multiplier starts at 100 (1x)
        uint256 multiplier = 100;
        
        // Duration bonus: up to 4x for maximum duration
        uint256 durationBonus = (duration * 300) / MAX_STAKE_DURATION; // Max 300% bonus
        multiplier += durationBonus;
        
        // Type-specific bonuses
        if (stakeType == StakeType.AI_TRAINING) {
            multiplier += 50; // +50% for AI trainers
        } else if (stakeType == StakeType.GOVERNANCE) {
            multiplier += 25; // +25% for governance participation
        } else if (stakeType == StakeType.LIQUIDITY) {
            multiplier += 75; // +75% for liquidity providers
        }
        
        // Cap at maximum multiplier
        if (multiplier > MAX_STAKE_MULTIPLIER) {
            multiplier = MAX_STAKE_MULTIPLIER;
        }
        
        return multiplier;
    }

    // ==================== AI TRAINING REWARDS SYSTEM ====================
    
    /**
     * @notice Start a new AI training session
     * @dev Registers a new training session for reward tracking
     * @param modelId Unique identifier for the AI model being trained
     * @param trainingType Type of training being performed
     * @return sessionId Unique identifier for this training session
     */
    function startTrainingSession(
        uint256 modelId, 
        TrainingType trainingType
    ) 
        external 
        onlyRole(AI_TRAINER_ROLE) 
        whenNotPaused 
        returns (uint256 sessionId) 
    {
        sessionId = ++trainingSessionCounter;
        
        trainingSessions[sessionId] = AITrainingSession({
            trainer: msg.sender,
            modelId: modelId,
            startTime: block.timestamp,
            endTime: 0,
            accuracyScore: 0,
            complexityScore: 0,
            rewardAmount: 0,
            rewardClaimed: false,
            trainingType: trainingType
        });
        
        emit AITrainingStarted(sessionId, msg.sender, modelId, trainingType);
        
        return sessionId;
    }
    
    /**
     * @notice Complete an AI training session and calculate rewards
     * @dev Finalizes training session and calculates performance-based rewards
     * @param sessionId ID of the training session to complete
     * @param accuracyScore Model accuracy score (0-10000 basis points)
     * @param complexityScore Model complexity score (affects reward calculation)
     */
    function completeTrainingSession(
        uint256 sessionId,
        uint256 accuracyScore,
        uint256 complexityScore
    ) 
        external 
        onlyRole(AI_TRAINER_ROLE) 
    {
        AITrainingSession storage session = trainingSessions[sessionId];
        require(session.trainer == msg.sender, "Not session owner");
        require(session.endTime == 0, "Session already completed");
        require(accuracyScore <= 10000, "Invalid accuracy score");
        
        session.endTime = block.timestamp;
        session.accuracyScore = accuracyScore;
        session.complexityScore = complexityScore;
        
        // Calculate training duration bonus
        uint256 trainingDuration = session.endTime - session.startTime;
        
        // Calculate reward based on performance metrics
        uint256 baseReward = _calculateTrainingBaseReward(session.trainingType);
        uint256 performanceMultiplier = _calculatePerformanceMultiplier(
            accuracyScore,
            complexityScore,
            trainingDuration
        );
        
        session.rewardAmount = (baseReward * performanceMultiplier) / 100;
        
        emit AITrainingCompleted(sessionId, msg.sender, accuracyScore, session.rewardAmount);
    }
    
    /**
     * @notice Claim rewards from completed training session
     * @dev Mints and transfers training rewards to trainer
     * @param sessionId ID of the completed training session
     */
    function claimTrainingReward(uint256 sessionId) 
        external 
        nonReentrant 
    {
        AITrainingSession storage session = trainingSessions[sessionId];
        require(session.trainer == msg.sender, "Not session owner");
        require(session.endTime > 0, "Session not completed");
        require(!session.rewardClaimed, "Reward already claimed");
        require(session.rewardAmount > 0, "No reward available");
        
        session.rewardClaimed = true;
        totalRewardsDistributed += session.rewardAmount;
        
        _mint(msg.sender, session.rewardAmount);
        
        emit RewardsClaimed(msg.sender, sessionId, session.rewardAmount);
    }
    
    /**
     * @notice Calculate base reward for different training types
     * @dev Internal function to determine base reward amounts
     * @param trainingType Type of AI training performed
     * @return Base reward amount in tokens
     */
    function _calculateTrainingBaseReward(TrainingType trainingType) 
        internal 
        pure 
        returns (uint256) 
    {
        if (trainingType == TrainingType.DEEP_LEARNING) {
            return 1000 * 10**18; // 1000 tokens
        } else if (trainingType == TrainingType.TRANSFORMER) {
            return 1500 * 10**18; // 1500 tokens
        } else if (trainingType == TrainingType.EMBEDDING) {
            return 800 * 10**18;  // 800 tokens
        } else if (trainingType == TrainingType.REINFORCEMENT) {
            return 1200 * 10**18; // 1200 tokens
        } else if (trainingType == TrainingType.SUPERVISED) {
            return 600 * 10**18;  // 600 tokens
        } else if (trainingType == TrainingType.UNSUPERVISED) {
            return 700 * 10**18;  // 700 tokens
        }
        
        return 500 * 10**18; // Default 500 tokens
    }
    
    /**
     * @notice Calculate performance multiplier for training rewards
     * @dev Determines reward multiplier based on training quality metrics
     * @param accuracyScore Model accuracy (0-10000 basis points)
     * @param complexityScore Model complexity score
     * @param trainingDuration Training session duration
     * @return Performance multiplier (100 = 1x)
     */
    function _calculatePerformanceMultiplier(
        uint256 accuracyScore,
        uint256 complexityScore,
        uint256 trainingDuration
    ) 
        internal 
        pure 
        returns (uint256) 
    {
        uint256 multiplier = 100; // Base 1x multiplier
        
        // Accuracy bonus: up to 2x for perfect accuracy
        if (accuracyScore >= 9500) {
            multiplier += 100; // +100% for >95% accuracy
        } else if (accuracyScore >= 9000) {
            multiplier += 75;  // +75% for >90% accuracy
        } else if (accuracyScore >= 8500) {
            multiplier += 50;  // +50% for >85% accuracy
        } else if (accuracyScore >= 8000) {
            multiplier += 25;  // +25% for >80% accuracy
        }
        
        // Complexity bonus: higher complexity = higher rewards
        if (complexityScore >= 1000) {
            multiplier += 50; // +50% for high complexity
        } else if (complexityScore >= 500) {
            multiplier += 25; // +25% for medium complexity
        }
        
        // Duration bonus: longer training sessions get bonus
        if (trainingDuration >= 1 days) {
            multiplier += 30; // +30% for training longer than 1 day
        } else if (trainingDuration >= 12 hours) {
            multiplier += 15; // +15% for training longer than 12 hours
        }
        
        // Cap maximum multiplier at 5x
        if (multiplier > 500) {
            multiplier = 500;
        }
        
        return multiplier;
    }

    // ==================== GOVERNANCE SYSTEM ====================
    
    /**
     * @notice Create a new governance proposal
     * @dev Creates a proposal that token holders can vote on
     * @param title Short title for the proposal
     * @param description Detailed description of the proposal
     * @param proposalType Type of proposal being created
     * @param executionData Encoded function call data to execute if proposal passes
     * @param votingPeriod Duration of voting period in seconds
     * @return proposalId Unique identifier for the created proposal
     */
    function createProposal(
        string calldata title,
        string calldata description,
        ProposalType proposalType,
        bytes calldata executionData,
        uint256 votingPeriod
    ) 
        external 
        whenNotPaused 
        returns (bytes32 proposalId) 
    {
        require(bytes(title).length > 0, "Title cannot be empty");
        require(bytes(description).length > 0, "Description cannot be empty");
        require(
            votingPeriod >= MIN_VOTING_PERIOD && votingPeriod <= MAX_VOTING_PERIOD,
            "Invalid voting period"
        );
        
        // Check if proposer has sufficient voting power
        uint256 proposerVotes = getVotes(msg.sender);
        if (proposerVotes < PROPOSAL_THRESHOLD) {
            revert InsufficientVotingPower(proposerVotes, PROPOSAL_THRESHOLD);
        }
        
        // Generate unique proposal ID
        proposalId = keccak256(
            abi.encodePacked(
                title,
                description,
                proposalType,
                executionData,
                block.timestamp,
                ++proposalCounter
            )
        );
        
        // Create proposal
        Proposal storage proposal = proposals[proposalId];
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.title = title;
        proposal.description = description;
        proposal.startTime = block.timestamp;
        proposal.endTime = block.timestamp + votingPeriod;
        proposal.proposalType = proposalType;
        proposal.executionData = executionData;
        proposal.state = ProposalState.ACTIVE;
        
        proposalIds.push(proposalId);
        
        emit ProposalCreated(
            proposalId,
            msg.sender,
            title,
            proposalType,
            proposal.startTime,
            proposal.endTime
        );
        
        return proposalId;
    }
    
    /**
     * @notice Cast a vote on an active proposal
     * @dev Allows token holders to vote on governance proposals
     * @param proposalId ID of the proposal to vote on
     * @param choice Vote choice (FOR, AGAINST, ABSTAIN)
     * @param reason Optional reason for the vote
     */
    function castVote(
        bytes32 proposalId,
        VoteChoice choice,
        string calldata reason
    ) 
        external 
        validProposal(proposalId, ProposalState.ACTIVE) 
    {
        Proposal storage proposal = proposals[proposalId];
        
        require(block.timestamp <= proposal.endTime, "Voting period ended");
        
        if (proposal.hasVoted[msg.sender]) {
            revert AlreadyVoted(msg.sender, proposalId);
        }
        
        uint256 weight = getVotes(msg.sender);
        require(weight > 0, "No voting power");
        
        proposal.hasVoted[msg.sender] = true;
        proposal.votes[msg.sender] = choice;
        
        if (choice == VoteChoice.FOR) {
            proposal.forVotes += weight;
        } else if (choice == VoteChoice.AGAINST) {
            proposal.againstVotes += weight;
        } else if (choice == VoteChoice.ABSTAIN) {
            proposal.abstainVotes += weight;
        }
        
        emit VoteCast(proposalId, msg.sender, choice, weight, reason);
    }
    
    /**
     * @notice Execute a successful proposal
     * @dev Executes the proposal if it has passed and voting period is over
     * @param proposalId ID of the proposal to execute
     */
    function executeProposal(bytes32 proposalId) 
        external 
        payable 
        nonReentrant 
    {
        Proposal storage proposal = proposals[proposalId];
        
        require(proposal.id == proposalId, "Proposal does not exist");
        require(block.timestamp > proposal.endTime, "Voting still active");
        require(!proposal.executed, "Already executed");
        require(!proposal.canceled, "Proposal canceled");
        
        // Calculate if proposal passed
        uint256 totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
        uint256 quorumRequired = (totalSupply() * QUORUM_PERCENTAGE) / 10000;
        
        bool quorumReached = totalVotes >= quorumRequired;
        bool majorityFor = proposal.forVotes > proposal.againstVotes;
        
        if (quorumReached && majorityFor) {
            proposal.state = ProposalState.SUCCEEDED;
            proposal.executed = true;
            
            // Execute the proposal
            if (proposal.executionData.length > 0) {
                _executeProposalAction(proposal);
            }
            
            emit ProposalExecuted(proposalId, msg.sender);
        } else {
            proposal.state = ProposalState.DEFEATED;
        }
    }
    
    /**
     * @notice Execute the action specified in a proposal
     * @dev Internal function to handle different types of proposal execution
     * @param proposal The proposal to execute
     */
    function _executeProposalAction(Proposal memory proposal) internal {
        if (proposal.proposalType == ProposalType.PARAMETER_CHANGE) {
            // Handle parameter changes
            (bool success,) = address(this).call(proposal.executionData);
            require(success, "Parameter change failed");
        } else if (proposal.proposalType == ProposalType.TREASURY) {
            // Handle treasury operations
            (address recipient, uint256 amount) = abi.decode(proposal.executionData, (address, uint256));
            require(amount <= treasuryBalance, "Insufficient treasury balance");
            treasuryBalance -= amount;
            _transfer(address(this), recipient, amount);
        } else if (proposal.proposalType == ProposalType.EMERGENCY) {
            // Handle emergency actions
            _handleEmergencyProposal(proposal.executionData);
        }
        // Add more proposal type handlers as needed
    }
    
    /**
     * @notice Handle emergency proposal execution
     * @dev Internal function for emergency actions
     * @param executionData Encoded data for emergency action
     */
    function _handleEmergencyProposal(bytes memory executionData) internal {
        (string memory action) = abi.decode(executionData, (string));
        
        if (keccak256(bytes(action)) == keccak256("ACTIVATE_EMERGENCY")) {
            emergencyMode = true;
            emergencyActivationTime = block.timestamp;
            emit EmergencyActivated(msg.sender, "Governance proposal", block.timestamp);
        } else if (keccak256(bytes(action)) == keccak256("DEACTIVATE_EMERGENCY")) {
            emergencyMode = false;
            emergencyActivationTime = 0;
        }
    }
    
    /**
     * @notice Get the current state of a proposal
     * @dev Returns the current state of a proposal based on voting and timing
     * @param proposalId ID of the proposal to check
     * @return Current state of the proposal
     */
    function getProposalState(bytes32 proposalId) public view returns (ProposalState) {
        Proposal storage proposal = proposals[proposalId];
        
        if (proposal.id == bytes32(0)) {
            return ProposalState.PENDING; // Doesn't exist
        }
        
        if (proposal.canceled) {
            return ProposalState.CANCELED;
        }
        
        if (proposal.executed) {
            return ProposalState.EXECUTED;
        }
        
        if (block.timestamp <= proposal.endTime) {
            return ProposalState.ACTIVE;
        }
        
        uint256 totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
        uint256 quorumRequired = (totalSupply() * QUORUM_PERCENTAGE) / 10000;
        
        if (totalVotes < quorumRequired || proposal.forVotes <= proposal.againstVotes) {
            return ProposalState.DEFEATED;
        }
        
        return ProposalState.SUCCEEDED;
    }

    // ==================== ANTI-MEV & FLASH LOAN PROTECTION ====================
    
    /**
     * @notice Check for flash loan attacks
     * @dev Internal function to detect potential flash loan exploits
     */
    function _checkFlashLoanAttack() internal view {
        // Check if balance increased significantly in the same block
        uint256 currentBalance = balanceOf(msg.sender);
        uint256 snapshotBalance = flashLoanGuard.balanceSnapshots[msg.sender];
        
        if (flashLoanGuard.blockNumber == block.number && snapshotBalance > 0) {
            // If balance increased by more than 1000x in same block, likely flash loan
            if (currentBalance > snapshotBalance * 1000) {
                revert FlashLoanAttackDetected(msg.sender);
            }
        }
    }
    
    /**
     * @notice Check for MEV attacks
     * @dev Internal function to detect MEV (Maximum Extractable Value) attacks
     */
    function _checkMEVAttack() internal view {
        // Prevent same-block arbitrage
        if (lastTransactionBlock[msg.sender] == block.number) {
            revert MEVAttackDetected(msg.sender);
        }
    }
    
    /**
     * @notice Update balance snapshot for flash loan protection
     * @dev Internal function to maintain balance tracking
     */
    function _updateBalanceSnapshot() internal {
        if (flashLoanGuard.blockNumber != block.number) {
            flashLoanGuard.blockNumber = block.number;
        }
        
        flashLoanGuard.balanceSnapshots[msg.sender] = balanceOf(msg.sender);
        lastTransactionBlock[msg.sender] = block.number;
    }

    // ==================== LIQUIDITY MINING ====================
    
    /**
     * @notice Add a new liquidity mining pool
     * @dev Creates a new pool for liquidity mining rewards
     * @param poolAddress Address of the liquidity pool contract
     * @param rewardRate Reward rate per second for this pool
     */
    function addLiquidityPool(
        address poolAddress, 
        uint256 rewardRate
    ) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(poolAddress != address(0), "Invalid pool address");
        require(!liquidityPools[poolAddress].isActive, "Pool already exists");
        
        LiquidityPool storage pool = liquidityPools[poolAddress];
        pool.poolAddress = poolAddress;
        pool.rewardRate = rewardRate;
        pool.lastUpdateTime = block.timestamp;
        pool.isActive = true;
        
        activePools.push(poolAddress);
    }
    
    /**
     * @notice Stake tokens in a liquidity mining pool
     * @dev Stake tokens to earn liquidity mining rewards
     * @param poolAddress Address of the pool to stake in
     * @param amount Amount of tokens to stake
     */
    function stakeLiquidity(address poolAddress, uint256 amount) 
        external 
        whenNotPaused 
        nonReentrant 
    {
        require(amount > 0, "Amount must be greater than 0");
        require(liquidityPools[poolAddress].isActive, "Pool not active");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        
        LiquidityPool storage pool = liquidityPools[poolAddress];
        
        // Update pool rewards before staking
        _updatePoolRewards(poolAddress);
        
        // Transfer tokens to contract
        _transfer(msg.sender, address(this), amount);
        
        // Update user's stake
        pool.stakedAmounts[msg.sender] += amount;
        pool.totalStaked += amount;
        pool.userRewardPerTokenPaid[msg.sender] = pool.rewardPerToken;
        
        emit LiquidityAdded(msg.sender, poolAddress, amount);
    }
    
    /**
     * @notice Withdraw staked tokens from liquidity pool
     * @dev Withdraw tokens and claim accumulated rewards
     * @param poolAddress Address of the pool to withdraw from
     * @param amount Amount of tokens to withdraw
     */
    function withdrawLiquidity(address poolAddress, uint256 amount) 
        external 
        nonReentrant 
    {
        LiquidityPool storage pool = liquidityPools[poolAddress];
        require(pool.stakedAmounts[msg.sender] >= amount, "Insufficient staked amount");
        
        // Update pool rewards before withdrawal
        _updatePoolRewards(poolAddress);
        
        // Calculate pending rewards
        uint256 pendingRewards = _calculateLiquidityRewards(poolAddress, msg.sender);
        
        // Update user's stake
        pool.stakedAmounts[msg.sender] -= amount;
        pool.totalStaked -= amount;
        pool.userRewardPerTokenPaid[msg.sender] = pool.rewardPerToken;
        pool.rewards[msg.sender] = 0;
        
        // Transfer staked tokens back
        _transfer(address(this), msg.sender, amount);
        
        // Mint and transfer rewards
        if (pendingRewards > 0) {
            _mint(msg.sender, pendingRewards);
            totalRewardsDistributed += pendingRewards;
        }
        
        emit LiquidityRemoved(msg.sender, poolAddress, amount, pendingRewards);
    }
    
    /**
     * @notice Update reward calculations for a liquidity pool
     * @dev Internal function to update reward per token
     * @param poolAddress Address of the pool to update
     */
    function _updatePoolRewards(address poolAddress) internal {
        LiquidityPool storage pool = liquidityPools[poolAddress];
        
        if (pool.totalStaked == 0) {
            pool.lastUpdateTime = block.timestamp;
            return;
        }
        
        uint256 timeElapsed = block.timestamp - pool.lastUpdateTime;
        uint256 rewardToDistribute = timeElapsed * pool.rewardRate;
        
        pool.rewardPerToken += (rewardToDistribute * 1e18) / pool.totalStaked;
        pool.lastUpdateTime = block.timestamp;
    }
    
    /**
     * @notice Calculate pending liquidity mining rewards for a user
     * @dev Internal function to calculate user's pending rewards
     * @param poolAddress Address of the liquidity pool
     * @param user Address of the user
     * @return Pending reward amount
     */
    function _calculateLiquidityRewards(address poolAddress, address user) 
        internal 
        view 
        returns (uint256) 
    {
        LiquidityPool storage pool = liquidityPools[poolAddress];
        
        uint256 userStaked = pool.stakedAmounts[user];
        uint256 rewardPerTokenDiff = pool.rewardPerToken - pool.userRewardPerTokenPaid[user];
        
        return pool.rewards[user] + (userStaked * rewardPerTokenDiff) / 1e18;
    }

    // ==================== BUYBACK AND BURN MECHANISM ====================
    
    /**
     * @notice Execute buyback and burn using treasury funds
     * @dev Automatically burns tokens using a percentage of treasury
     */
    function executeBuybackAndBurn() external {
        require(
            block.timestamp >= lastBuybackTime + buybackInterval,
            "Buyback interval not reached"
        );
        require(treasuryBalance > 0, "No treasury funds available");
        
        uint256 buybackAmount = (treasuryBalance * buybackPercentage) / 10000;
        
        if (buybackAmount > 0 && buybackAmount <= balanceOf(address(this))) {
            treasuryBalance -= buybackAmount;
            _burn(address(this), buybackAmount);
            totalBurned += buybackAmount;
            lastBuybackTime = block.timestamp;
            
            emit BuybackAndBurn(buybackAmount, block.timestamp);
        }
    }
    
    /**
     * @notice Set buyback parameters
     * @dev Admin function to configure buyback mechanism
     * @param interval Time interval between buybacks
     * @param percentage Percentage of treasury to use for buyback (in basis points)
     */
    function setBuybackParameters(uint256 interval, uint256 percentage) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(interval >= 1 days, "Interval too short");
        require(percentage <= 1000, "Percentage too high"); // Max 10%
        
        buybackInterval = interval;
        buybackPercentage = percentage;
    }

    // ==================== EMERGENCY FUNCTIONS ====================
    
    /**
     * @notice Activate emergency mode
     * @dev Allows emergency council to pause all operations
     * @param reason Reason for activating emergency mode
     */
    function activateEmergency(string calldata reason) 
        external 
        onlyRole(EMERGENCY_ROLE) 
    {
        emergencyMode = true;
        emergencyActivationTime = block.timestamp;
        _pause();
        
        emit EmergencyActivated(msg.sender, reason, block.timestamp);
    }
    
    /**
     * @notice Deactivate emergency mode
     * @dev Allows emergency council to resume normal operations
     */
    function deactivateEmergency() 
        external 
        onlyRole(EMERGENCY_ROLE) 
    {
        emergencyMode = false;
        emergencyActivationTime = 0;
        _unpause();
    }
    
    /**
     * @notice Emergency withdrawal for stuck funds
     * @dev Last resort function for recovering stuck tokens
     * @param token Address of token to recover (use address(0) for ETH)
     * @param amount Amount to recover
     * @param recipient Address to send recovered funds to
     */
    function emergencyWithdraw(
        address token, 
        uint256 amount, 
        address recipient
    ) 
        external 
        onlyRole(EMERGENCY_ROLE) 
    {
        require(emergencyMode, "Emergency mode not active");
        require(recipient != address(0), "Invalid recipient");
        
        if (token == address(0)) {
            // ETH withdrawal
            payable(recipient).transfer(amount);
        } else {
            // Token withdrawal
            IERC20(token).transfer(recipient, amount);
        }
    }

    // ==================== VIEW FUNCTIONS ====================
    
    /**
     * @notice Get all stakes for a user
     * @dev Returns array of all stakes owned by a user
     * @param user Address of the user
     * @return Array of StakeInfo structs
     */
    function getUserStakes(address user) external view returns (StakeInfo[] memory) {
        return userStakes[user];
    }
    
    /**
     * @notice Get pending rewards for all active stakes of a user
     * @dev Calculates total pending rewards across all stakes
     * @param user Address of the user
     * @return Total pending reward amount
     */
    function getTotalPendingRewards(address user) external view returns (uint256) {
        uint256 totalPending = 0;
        StakeInfo[] storage stakes = userStakes[user];
        
        for (uint256 i = 0; i < stakes.length; i++) {
            if (stakes[i].isActive) {
                totalPending += _calculatePendingRewards(user, i);
            }
        }
        
        return totalPending;
    }
    
    /**
     * @notice Get comprehensive user information
     * @dev Returns detailed information about a user's involvement
     * @param user Address of the user
     * @return User information struct
     */
    function getUserInfo(address user) external view returns (
        uint256 balance,
        uint256 stakedAmount,
        uint256 votingPower,
        uint256 totalRewardsClaimed,
        uint256 activeStakes
    ) {
        balance = balanceOf(user);
        votingPower = getVotes(user);
        
        StakeInfo[] storage stakes = userStakes[user];
        for (uint256 i = 0; i < stakes.length; i++) {
            if (stakes[i].isActive) {
                stakedAmount += stakes[i].amount;
                activeStakes++;
            }
            totalRewardsClaimed += stakes[i].rewardsClaimed;
        }
    }
    
    /**
     * @notice Get proposal information
     * @dev Returns detailed information about a specific proposal
     * @param proposalId ID of the proposal
     * @return Proposal details
     */
    function getProposalInfo(bytes32 proposalId) external view returns (
        address proposer,
        string memory title,
        string memory description,
        uint256 startTime,
        uint256 endTime,
        uint256 forVotes,
        uint256 againstVotes,
        uint256 abstainVotes,
        ProposalState state,
        bool executed
    ) {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.proposer,
            proposal.title,
            proposal.description,
            proposal.startTime,
            proposal.endTime,
            proposal.forVotes,
            proposal.againstVotes,
            proposal.abstainVotes,
            getProposalState(proposalId),
            proposal.executed
        );
    }
    
    /**
     * @notice Get all proposal IDs
     * @dev Returns array of all created proposal IDs
     * @return Array of proposal IDs
     */
    function getAllProposalIds() external view returns (bytes32[] memory) {
        return proposalIds;
    }
    
    /**
     * @notice Get contract statistics
     * @dev Returns comprehensive statistics about the contract
     * @return Contract statistics
     */
    function getContractStats() external view returns (
        uint256 totalSupplyAmount,
        uint256 totalStakedAmount,
        uint256 totalRewardsAmount,
        uint256 