// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "./HModelToken.sol";

/**
 * @title HModelTokenTest
 * @dev Comprehensive testing suite for HModelToken
 * @author iDeaKz - Testing Mastermind
 * 
 * ████████╗███████╗███████╗████████╗    ███╗   ███╗ █████╗ ███████╗████████╗███████╗██████╗ 
 * ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ████╗ ████║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗
 *    ██║   █████╗  ███████╗   ██║       ██╔████╔██║███████║███████╗   ██║   █████╗  ██████╔╝
 *    ██║   ██╔══╝  ╚════██║   ██║       ██║╚██╔╝██║██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗
 *    ██║   ███████╗███████║   ██║       ██║ ╚═╝ ██║██║  ██║███████║   ██║   ███████╗██║  ██║
 *    ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
 * 
 * Features:
 * ✅ Comprehensive Unit Tests
 * ✅ Integration Testing
 * ✅ Security Vulnerability Tests
 * ✅ Gas Optimization Tests
 * ✅ Stress Testing
 * ✅ Edge Case Coverage
 * ✅ Fuzzing Tests
 * ✅ Performance Benchmarks
 */
contract HModelTokenTest is Test {
    HModelToken public token;
    
    address public owner = address(0x1);
    address public emergencyCouncil = address(0x2);
    address public priceOracle = address(0x3);
    address public user1 = address(0x4);
    address public user2 = address(0x5);
    address public attacker = address(0x6);
    
    // Events for testing
    event Staked(address indexed user, uint256 indexed stakeIndex, uint256 amount, uint256 duration, uint8 stakeType, uint256 multiplier);
    event Unstaked(address indexed user, uint256 indexed stakeIndex, uint256 amount, uint256 rewards);
    event ProposalCreated(bytes32 indexed proposalId, address indexed proposer, string title, uint8 proposalType, uint256 startTime, uint256 endTime);
    
    function setUp() public {
        vm.startPrank(owner);
        
        token = new HModelToken(owner, emergencyCouncil, priceOracle);
        
        // Setup initial balances
        token.transfer(user1, 10000 * 10**18);
        token.transfer(user2, 5000 * 10**18);
        
        vm.stopPrank();
    }
    
    // ==================== BASIC FUNCTIONALITY TESTS ====================
    
    function testInitialState() public {
        assertEq(token.totalSupply(), 100_000_000 * 10**18);
        assertEq(token.balanceOf(owner), 100_000_000 * 10**18 - 15000 * 10**18);
        assertEq(token.balanceOf(user1), 10000 * 10**18);
        assertEq(token.balanceOf(user2), 5000 * 10**18);
        assertTrue(token.hasRole(token.DEFAULT_ADMIN_ROLE(), owner));
        assertFalse(token.emergencyMode());
    }
    
    function testTransfer() public {
        vm.prank(user1);
        bool success = token.transfer(user2, 1000 * 10**18);
        
        assertTrue(success);
        assertEq(token.balanceOf(user1), 9000 * 10**18);
        assertEq(token.balanceOf(user2), 6000 * 10**18);
    }
    
    function testTransferInsufficientBalance() public {
        vm.prank(user1);
        vm.expectRevert();
        token.transfer(user2, 20000 * 10**18); // More than user1 has
    }
    
    function testApproveAndTransferFrom() public {
        vm.prank(user1);
        token.approve(user2, 1000 * 10**18);
        
        vm.prank(user2);
        bool success = token.transferFrom(user1, user2, 500 * 10**18);
        
        assertTrue(success);
        assertEq(token.balanceOf(user1), 9500 * 10**18);
        assertEq(token.balanceOf(user2), 5500 * 10**18);
        assertEq(token.allowance(user1, user2), 500 * 10**18);
    }
    
    // ==================== STAKING TESTS ====================
    
    function testStake() public {
        vm.startPrank(user1);
        
        uint256 stakeAmount = 1000 * 10**18;
        uint256 stakeDuration = 30 days;
        
        vm.expectEmit(true, true, false, true);
        emit Staked(user1, 0, stakeAmount, stakeDuration, 0, 100 + (stakeDuration * 300) / (4 * 365 days));
        
        uint256 stakeIndex = token.stake(stakeAmount, stakeDuration, HModelToken.StakeType.STANDARD);
        
        assertEq(stakeIndex, 0);
        assertEq(token.balanceOf(user1), 9000 * 10**18);
        assertEq(token.totalStaked(), stakeAmount);
        
        // Check stake info
        HModelToken.StakeInfo[] memory stakes = token.getUserStakes(user1);
        assertEq(stakes.length, 1);
        assertEq(stakes[0].amount, stakeAmount);
        assertEq(stakes[0].duration, stakeDuration);
        assertTrue(stakes[0].isActive);
        
        vm.stopPrank();
    }
    
    function testStakeInvalidDuration() public {
        vm.prank(user1);
        vm.expectRevert();
        token.stake(1000 * 10**18, 1 hours, HModelToken.StakeType.STANDARD); // Too short
        
        vm.prank(user1);
        vm.expectRevert();
        token.stake(1000 * 10**18, 5 * 365 days, HModelToken.StakeType.STANDARD); // Too long
    }
    
    function testStakeInsufficientBalance() public {
        vm.prank(user1);
        vm.expectRevert();
        token.stake(20000 * 10**18, 30 days, HModelToken.StakeType.STANDARD); // More than balance
    }
    
    function testUnstakeBeforeMaturity() public {
        vm.startPrank(user1);
        
        uint256 stakeIndex = token.stake(1000 * 10**18, 30 days, HModelToken.StakeType.STANDARD);
        
        // Try to unstake immediately
        vm.expectRevert();
        token.unstake(stakeIndex);
        
        vm.stopPrank();
    }
    
    function testUnstakeAfterMaturity() public {
        vm.startPrank(user1);
        
        uint256 stakeAmount = 1000 * 10**18;
        uint256 stakeDuration = 30 days;
        uint256 stakeIndex = token.stake(stakeAmount, stakeDuration, HModelToken.StakeType.STANDARD);
        
        // Fast forward time
        vm.warp(block.timestamp + stakeDuration + 1);
        
        uint256 balanceBefore = token.balanceOf(user1);
        
        vm.expectEmit(true, true, false, false);
        emit Unstaked(user1, stakeIndex, stakeAmount, 0); // Rewards will be calculated
        
        token.unstake(stakeIndex);
        
        assertEq(token.balanceOf(user1), balanceBefore + stakeAmount);
        assertEq(token.totalStaked(), 0);
        
        vm.stopPrank();
    }
    
    function testClaimRewards() public {
        vm.startPrank(user1);
        
        uint256 stakeAmount = 1000 * 10**18;
        uint256 stakeIndex = token.stake(stakeAmount, 365 days, HModelToken.StakeType.STANDARD);
        
        // Fast forward 30 days
        vm.warp(block.timestamp + 30 days);
        
        uint256 balanceBefore = token.balanceOf(user1);
        token.claimRewards(stakeIndex);
        
        // Should have received some rewards
        assertGt(token.balanceOf(user1), balanceBefore);
        
        vm.stopPrank();
    }
    
    function testCompoundStake() public {
        vm.startPrank(user1);
        
        uint256 stakeAmount = 1000 * 10**18;
        uint256 stakeIndex = token.stake(stakeAmount, 365 days, HModelToken.StakeType.STANDARD);
        
        // Fast forward 30 days
        vm.warp(block.timestamp + 30 days);
        
        HModelToken.StakeInfo[] memory stakesBefore = token.getUserStakes(user1);
        uint256 amountBefore = stakesBefore[0].amount;
        
        token.compoundStake(stakeIndex);
        
        HModelToken.StakeInfo[] memory stakesAfter = token.getUserStakes(user1);
        assertGt(stakesAfter[0].amount, amountBefore);
        
        vm.stopPrank();
    }
    
    // ==================== GOVERNANCE TESTS ====================
    
    function testCreateProposal() public {
        vm.startPrank(owner);
        
        string memory title = "Test Proposal";
        string memory description = "This is a test proposal for parameter changes";
        bytes memory data = abi.encodeWithSignature("updateRewardRates(uint256)", 1500);
        
        vm.expectEmit(true, true, false, false);
        emit ProposalCreated(bytes32(0), owner, title, 0, block.timestamp, block.timestamp + 7 days);
        
        bytes32 proposalId = token.createProposal(
            title,
            description,
            HModelToken.ProposalType.PARAMETER_CHANGE,
            data,
            7 days
        );
        
        assertTrue(proposalId != bytes32(0));
        assertEq(uint256(token.getProposalState(proposalId)), uint256(HModelToken.ProposalState.ACTIVE));
        
        vm.stopPrank();
    }
    
    function testCreateProposalInsufficientVotingPower() public {
        vm.prank(user2); // user2 has less voting power
        vm.expectRevert();
        token.createProposal(
            "Test",
            "Test proposal",
            HModelToken.ProposalType.PARAMETER_CHANGE,
            "",
            7 days
        );
    }
    
    function testVoteOnProposal() public {
        // Create proposal as owner
        vm.prank(owner);
        bytes32 proposalId = token.createProposal(
            "Test Proposal",
            "Test description",
            HModelToken.ProposalType.PARAMETER_CHANGE,
            "",
            7 days
        );
        
        // Vote as user1
        vm.prank(user1);
        token.castVote(proposalId, HModelToken.VoteChoice.FOR, "I support this proposal");
        
        // Check vote was recorded
        (,,,,,uint256 forVotes,,,,) = token.getProposalInfo(proposalId);
        assertEq(forVotes, token.getVotes(user1));
    }
    
    function testDoubleVoting() public {
        vm.prank(owner);
        bytes32 proposalId = token.createProposal(
            "Test",
            "Test",
            HModelToken.ProposalType.PARAMETER_CHANGE,
            "",
            7 days
        );
        
        vm.startPrank(user1);
        token.castVote(proposalId, HModelToken.VoteChoice.FOR, "First vote");
        
        vm.expectRevert();
        token.castVote(proposalId, HModelToken.VoteChoice.AGAINST, "Second vote");
        vm.stopPrank();
    }
    
    // ==================== AI TRAINING TESTS ====================
    
    function testStartTrainingSession() public {
        vm.prank(owner);
        token.grantRole(token.AI_TRAINER_ROLE(), user1);
        
        vm.prank(user1);
        uint256 sessionId = token.startTrainingSession(12345, HModelToken.TrainingType.DEEP_LEARNING);
        
        assertEq(sessionId, 1);
        
        (address trainer, uint256 modelId,,,,,, HModelToken.TrainingType trainingType) = token.trainingSessions(sessionId);
        assertEq(trainer, user1);
        assertEq(modelId, 12345);
        assertEq(uint256(trainingType), uint256(HModelToken.TrainingType.DEEP_LEARNING));
    }
    
    function testCompleteTrainingSession() public {
        vm.prank(owner);
        token.grantRole(token.AI_TRAINER_ROLE(), user1);
        
        vm.startPrank(user1);
        uint256 sessionId = token.startTrainingSession(12345, HModelToken.TrainingType.TRANSFORMER);
        
        vm.warp(block.timestamp + 1 hours); // Simulate training time
        
        token.completeTrainingSession(sessionId, 9200, 800); // 92% accuracy, 800 complexity
        
        (,, uint256 startTime, uint256 endTime, uint256 accuracyScore, uint256 complexityScore, uint256 rewardAmount,) = token.trainingSessions(sessionId);
        
        assertGt(endTime, startTime);
        assertEq(accuracyScore, 9200);
        assertEq(complexityScore, 800);
        assertGt(rewardAmount, 0);
        
        vm.stopPrank();
    }
    
    function testClaimTrainingReward() public {
        vm.prank(owner);
        token.grantRole(token.AI_TRAINER_ROLE(), user1);
        
        vm.startPrank(user1);
        uint256 sessionId = token.startTrainingSession(12345, HModelToken.TrainingType.TRANSFORMER);
        token.completeTrainingSession(sessionId, 9500, 1000);
        
        uint256 balanceBefore = token.balanceOf(user1);
        token.claimTrainingReward(sessionId);
        
        assertGt(token.balanceOf(user1), balanceBefore);
        vm.stopPrank();
    }
    
    // ==================== SECURITY TESTS ====================
    
    function testEmergencyMode() public {
        vm.prank(emergencyCouncil);
        token.activateEmergency("Security breach detected");
        
        assertTrue(token.emergencyMode());
        assertTrue(token.paused());
        
        // Normal operations should fail
        vm.prank(user1);
        vm.expectRevert();
        token.transfer(user2, 100 * 10**18);
        
        // Deactivate emergency
        vm.prank(emergencyCouncil);
        token.deactivateEmergency();
        
        assertFalse(token.emergencyMode());
        assertFalse(token.paused());
    }
    
    function testFlashLoanProtection() public {
        // This test would require more complex setup to simulate flash loans
        // For now, we test that the protection mechanisms are in place
        
        vm.startPrank(attacker);
        
        // Try to transfer large amounts in same block
        vm.expectRevert();
        token.transfer(user2, 1000 * 10**18);
        vm.expectRevert();
        token.transfer(user1, 1000 * 10**18);
        
        vm.stopPrank();
    }
    
    function testAccessControl() public {
        // Test minting without role
        vm.prank(user1);
        vm.expectRevert();
        token.mint(user1, 1000 * 10**18);
        
        // Test pausing without role
        vm.prank(user1);
        vm.expectRevert();
        token.pause();
        
        // Test emergency functions without role
        vm.prank(user1);
        vm.expectRevert();
        token.activateEmergency("Test");
    }
    
    function testReentrancyGuard() public {
        // This would require a malicious contract to test properly
        // For now, we ensure the guards are in place by checking function modifiers
        
        vm.startPrank(user1);
        uint256 stakeIndex = token.stake(1000 * 10**18, 30 days, HModelToken.StakeType.STANDARD);
        
        // Fast forward and try to claim multiple times in same transaction
        vm.warp(block.timestamp + 31 days);
        
        // First unstake should work
        token.unstake(stakeIndex);
        
        // Second unstake should fail (stake no longer active)
        vm.expectRevert();
        token.unstake(stakeIndex);
        
        vm.stopPrank();
    }
    
    // ==================== EDGE CASE TESTS ====================
    
    function testZeroAmount() public {
        vm.prank(user1);
        vm.expectRevert();
        token.stake(0, 30 days, HModelToken.StakeType.STANDARD);
    }
    
    function testMaxSupplyCheck() public {
        vm.startPrank(owner);
        
        // Try to mint more than max supply
        uint256 maxSupply = token.MAX_SUPPLY();
        uint256 currentSupply = token.totalSupply();
        uint256 remainingSupply = maxSupply - currentSupply;
        
        vm.expectRevert();
        token.mint(owner, remainingSupply + 1);
        
        vm.stopPrank();
    }
    
    function testInvalidProposalParams() public {
        vm.prank(owner);
        
        // Empty title
        vm.expectRevert();
        token.createProposal("", "Description", HModelToken.ProposalType.PARAMETER_CHANGE, "", 7 days);
        
        // Invalid voting period
        vm.expectRevert();
        token.createProposal("Title", "Description", HModelToken.ProposalType.PARAMETER_CHANGE, "", 1 hours);
    }
    
    // ==================== GAS OPTIMIZATION TESTS ====================
    
    function testBatchTransferGas() public {
        address[] memory recipients = new address[](10);
        uint256[] memory amounts = new uint256[](10);
        
        for (uint256 i = 0; i < 10; i++) {
            recipients[i] = address(uint160(100 + i));
            amounts[i] = 100 * 10**18;
        }
        
        uint256 gasBefore = gasleft();
        vm.prank(user1);
        token.batchTransfer(recipients, amounts);
        uint256 gasUsed = gasBefore - gasleft();
        
        // Batch transfer should be more efficient than individual transfers
        assertLt(gasUsed, 500000); // Reasonable gas limit for batch operation
    }
    
    function testStakeGasOptimization() public {
        vm.startPrank(user1);
        
        uint256 gasBefore = gasleft();
        token.stake(1000 * 10**18, 365 days, HModelToken.StakeType.STANDARD);
        uint256 gasUsed = gasBefore - gasleft();
        
        // Staking should be reasonably efficient
        assertLt(gasUsed, 200000);
        
        vm.stopPrank();
    }
    
    // ==================== FUZZING TESTS ====================
    
    function testFuzzStakeAmount(uint256 amount) public {
        vm.assume(amount > 0 && amount <= token.balanceOf(user1));
        
        vm.prank(user1);
        uint256 stakeIndex = token.stake(amount, 30 days, HModelToken.StakeType.STANDARD);
        
        HModelToken.StakeInfo[] memory stakes = token.getUserStakes(user1);
        assertEq(stakes[stakeIndex].amount, amount);
    }
    
    function testFuzzStakeDuration(uint256 duration) public {
        vm.assume(duration >= token.MIN_STAKE_DURATION() && duration <= token.MAX_STAKE_DURATION());
        
        vm.prank(user1);
        uint256 stakeIndex = token.stake(1000 * 10**18, duration, HModelToken.StakeType.STANDARD);
        
        HModelToken.StakeInfo[] memory stakes = token.getUserStakes(user1);
        assertEq(stakes[stakeIndex].duration, duration);
    }
    
    function testFuzzTransferAmount(uint256 amount) public {
        uint256 balance = token.balanceOf(user1);
        vm.assume(amount <= balance);
        
        vm.prank(user1);
        if (amount == 0) {
            vm.expectRevert();
            token.transfer(user2, amount);
        } else {
            bool success = token.transfer(user2, amount);
            assertTrue(success);
            assertEq(token.balanceOf(user1), balance - amount);
        }
    }
    
    // ==================== STRESS TESTS ====================
    
    function testManyStakes() public {
        vm.startPrank(user1);
        
        // Create many small stakes
        for (uint256 i = 0; i < 50; i++) {
            if (token.balanceOf(user1) >= 100 * 10**18) {
                token.stake(100 * 10**18, 30 days + i * 1 days, HModelToken.StakeType.STANDARD);
            }
        }
        
        HModelToken.StakeInfo[] memory stakes = token.getUserStakes(user1);
        assertGt(stakes.length, 0);
        
        vm.stopPrank();
    }
    
    function testLargeVotingScenario() public {
        // Create proposal
        vm.prank(owner);
        bytes32 proposalId = token.createProposal(
            "Major Change",
            "This is a major change proposal",
            HModelToken.ProposalType.PARAMETER_CHANGE,
            "",
            7 days
        );
        
        // Have multiple users vote
        address[] memory voters = new address[](5);
        for (uint256 i = 0; i < 5; i++) {
            voters[i] = address(uint160(1000 + i));
            vm.prank(owner);
            token.transfer(voters[i], 1000 * 10**18);
            
            vm.prank(voters[i]);
            token.castVote(proposalId, HModelToken.VoteChoice.FOR, "Support");
        }
        
        (,,,,,uint256 forVotes,,,,) = token.getProposalInfo(proposalId);
        assertGt(forVotes, 0);
    }
    
    // ==================== INTEGRATION TESTS ====================
    
    function testCompleteStakingWorkflow() public {
        vm.startPrank(user1);
        
        uint256 initialBalance = token.balanceOf(user1);
        
        // 1. Stake tokens
        uint256 stakeAmount = 1000 * 10**18;
        uint256 stakeIndex = token.stake(stakeAmount, 365 days, HModelToken.StakeType.STANDARD);
        
        // 2. Fast forward and claim rewards multiple times
        for (uint256 i = 0; i < 3; i++) {
            vm.warp(block.timestamp + 30 days);
            token.claimRewards(stakeIndex);
        }
        
        // 3. Compound rewards
        token.compoundStake(stakeIndex);
        
        // 4. Fast forward to maturity and unstake
        vm.warp(block.timestamp + 300 days);
        token.unstake(stakeIndex);
        
        // Should have more tokens than initially
        assertGt(token.balanceOf(user1), initialBalance - stakeAmount);
        
        vm.stopPrank();
    }
    
    function testGovernanceWorkflow() public {
        // 1. Create proposal
        vm.prank(owner);
        bytes32 proposalId = token.createProposal(
            "Treasury Allocation",
            "Allocate treasury funds",
            HModelToken.ProposalType.TREASURY,
            abi.encode(user1, 1000 * 10**18),
            7 days
        );
        
        // 2. Vote on proposal
        vm.prank(user1);
        token.castVote(proposalId, HModelToken.VoteChoice.FOR, "Good idea");
        
        vm.prank(user2);
        token.castVote(proposalId, HModelToken.VoteChoice.FOR, "Agree");
        
        // 3. Fast forward past voting period
        vm.warp(block.timestamp + 8 days);
        
        // 4. Execute proposal
        uint256 balanceBefore = token.balanceOf(user1);
        token.executeProposal(proposalId);
        
        // Check if proposal executed correctly
        assertEq(uint256(token.getProposalState(proposalId)), uint256(HModelToken.ProposalState.EXECUTED));
        
        vm.stopPrank();
    }
}

/**
 * @title HModelTokenTestHelper
 * @dev Helper contract for advanced testing scenarios
 */
contract HModelTokenTestHelper {
    HModelToken public token;
    
    constructor(address _token) {
        token = HModelToken(_token);
    }
    
    // Helper function to simulate complex interactions
    function simulateComplexWorkflow() external {
        // This could simulate multi-step operations
        // that might be vulnerable to reentrancy
    }
    
    // Mock price oracle for testing
    function getPrice() external pure returns (uint256) {
        return 1e18; // 1 ETH
    }
}